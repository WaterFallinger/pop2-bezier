//! Native startup library. It renders metadata and local thumbnails only;
//! project creation and file operations remain explicit actions for the app.
use eframe::egui::{self, Align, Align2, Color32, FontId, Layout, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, TextureId, Vec2};
use std::{collections::{HashMap,HashSet},path::{Path,PathBuf},time::{Duration,Instant,SystemTime}};

const INK:Color32=Color32::from_rgb(34,43,62);
const MUTED:Color32=Color32::from_rgb(111,122,143);
const BLUE:Color32=Color32::from_rgb(58,110,242);
const LINE:Color32=Color32::from_rgb(228,233,241);
const CACHE_TARGET:usize=48;
const NAME_LIMIT:usize=120;

pub struct SaveEntryView<'a>{
    pub id:&'a str,
    pub name:&'a str,
    pub updated_label:&'a str,
    pub node_count:usize,
    /// One composite: canvas on the left, state tree on the right.
    pub preview_path:Option<&'a Path>,
}

#[derive(Clone,Debug,PartialEq,Eq)]
pub enum Action{
    Create{name:String},
    Open{id:String},
    Rename{id:String,name:String},
    Import,
    Settings,
    OpenLogs,
}

enum NameKind{Create,Rename{id:String}}
struct NameDialog{kind:NameKind,text:String,focus:bool,composing:bool}

pub struct SaveLibraryUi{
    dialog:Option<NameDialog>,
    previews:PreviewCache,
    #[cfg(test)]pub(crate) create_button_rect:Rect,
    #[cfg(test)]pub(crate) import_button_rect:Rect,
    #[cfg(test)]pub(crate) card_rects:Vec<(String,Rect)>,
    #[cfg(test)]pub(crate) rename_rects:Vec<(String,Rect)>,
    #[cfg(test)]pub(crate) name_input_rect:Rect,
    #[cfg(test)]pub(crate) confirm_rect:Rect,
    #[cfg(test)]pub(crate) cancel_rect:Rect,
}

impl Default for SaveLibraryUi{
    fn default()->Self{Self{dialog:None,previews:PreviewCache::default(),
        #[cfg(test)]create_button_rect:Rect::NOTHING,#[cfg(test)]import_button_rect:Rect::NOTHING,
        #[cfg(test)]card_rects:Vec::new(),#[cfg(test)]rename_rects:Vec::new(),
        #[cfg(test)]name_input_rect:Rect::NOTHING,#[cfg(test)]confirm_rect:Rect::NOTHING,#[cfg(test)]cancel_rect:Rect::NOTHING,
    }}
}

impl SaveLibraryUi{
    fn create_dialog(&mut self){self.dialog=Some(NameDialog{kind:NameKind::Create,text:String::new(),focus:true,composing:false});}
    fn rename_dialog(&mut self,id:&str,name:&str){self.dialog=Some(NameDialog{kind:NameKind::Rename{id:id.into()},text:name.into(),focus:true,composing:false});}

    pub fn show(&mut self,ctx:&egui::Context,entries:&[SaveEntryView<'_>],busy:bool,error:Option<&str>)->Vec<Action>{
        #[cfg(test)]{self.card_rects.clear();self.rename_rects.clear();self.create_button_rect=Rect::NOTHING;
            self.import_button_rect=Rect::NOTHING;self.name_input_rect=Rect::NOTHING;self.confirm_rect=Rect::NOTHING;self.cancel_rect=Rect::NOTHING;}
        self.previews.retain(entries);
        let mut actions=Vec::new();let modal=self.dialog.is_some();
        egui::TopBottomPanel::top("save_library_header").frame(egui::Frame::none().fill(Color32::WHITE)
            .inner_margin(egui::Margin::symmetric(32.,24.))).show(ctx,|ui|{
            ui.add_enabled_ui(!busy&&!modal,|ui|{
                ui.horizontal(|ui|{
                    ui.vertical(|ui|{
                        ui.label(RichText::new("存档").size(28.).strong().color(INK));
                        ui.add_space(5.);ui.label(RichText::new("选择一棵状态树，或新建存档开始创作").size(13.).color(MUTED));
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center),|ui|{
                        let create=ui.add(egui::Button::new(RichText::new("新建存档").color(Color32::WHITE).strong())
                            .fill(BLUE).stroke(Stroke::NONE).rounding(7.).min_size(Vec2::new(108.,38.)));
                        #[cfg(test)]{self.create_button_rect=create.rect;}
                        if create.clicked(){self.create_dialog();}
                        let import=ui.add(egui::Button::new("导入旧存档…").min_size(Vec2::new(108.,38.)));
                        #[cfg(test)]{self.import_button_rect=import.rect;}
                        if import.clicked(){actions.push(Action::Import);}
                        if ui.add(egui::Button::new("设置 / 权重下载").min_size(Vec2::new(128.,38.))).clicked(){actions.push(Action::Settings);}
                    });
                });
            });
        });
        egui::TopBottomPanel::bottom("save_library_footer").frame(egui::Frame::none()
            .fill(Color32::from_rgb(246,248,252)).inner_margin(egui::Margin::symmetric(32.,12.))).show(ctx,|ui|{
            ui.horizontal(|ui|{
                if busy{ui.spinner();ui.label(RichText::new("正在处理存档，请稍候…").color(MUTED));}
                else{ui.label(RichText::new(format!("{} 个存档",entries.len())).size(12.).color(MUTED));}
                ui.with_layout(Layout::right_to_left(Align::Center),|ui|{
                    if ui.add_enabled(!busy&&!modal,egui::Button::new("查看日志").frame(false)).clicked(){actions.push(Action::OpenLogs);}
                });
            });
        });
        egui::CentralPanel::default().frame(egui::Frame::none().fill(Color32::from_rgb(242,245,250))
            .inner_margin(egui::Margin::symmetric(32.,24.))).show(ctx,|ui|{
            if let Some(error)=error.filter(|text|!text.is_empty()){
                egui::Frame::none().fill(Color32::from_rgb(255,239,238)).rounding(7.).inner_margin(12.).show(ui,|ui|{
                    ui.label(RichText::new(error).color(Color32::from_rgb(168,58,55)));
                });ui.add_space(18.);
            }
            ui.add_enabled_ui(!busy&&!modal,|ui|{
                if entries.is_empty(){
                    ui.vertical_centered(|ui|{
                        ui.add_space((ui.available_height()*0.28).max(30.));
                        let(rect,_)=ui.allocate_exact_size(Vec2::new(82.,65.),Sense::hover());
                        paint_empty_icon(ui.painter(),rect);
                        ui.add_space(18.);ui.label(RichText::new("还没有存档").size(20.).strong().color(INK));
                        ui.add_space(8.);ui.label(RichText::new("新建存档后，就可以开始绘画。").size(13.).color(MUTED));
                    });
                }else{
                    egui::ScrollArea::vertical().id_source("save_library_cards").auto_shrink([false,false]).show(ui,|ui|{
                        let available=ui.available_width().max(100.);let gap=18.;
                        let columns=((available+gap)/(320.+gap)).floor().max(1.)as usize;
                        let width=((available-gap*(columns.saturating_sub(1))as f32)/columns as f32).max(80.);
                        for row in entries.chunks(columns){
                            ui.horizontal_top(|ui|{
                                ui.spacing_mut().item_spacing.x=gap;
                                for entry in row{self.card(ui,entry,width,&mut actions);}
                            });ui.add_space(gap);
                        }
                    });
                }
            });
        });
        self.show_name_dialog(ctx,entries,busy,&mut actions);
        if busy{ctx.request_repaint_after(Duration::from_millis(80));}
        actions
    }

    fn card(&mut self,ui:&mut egui::Ui,entry:&SaveEntryView<'_>,width:f32,actions:&mut Vec<Action>){
        let preview_height=(width-28.).max(40.)*0.52;
        let height=preview_height+112.;
        let(rect,_)=ui.allocate_exact_size(Vec2::new(width,height),Sense::hover());
        let open_rect=Rect::from_min_max(rect.min,Pos2::new(rect.right(),rect.bottom()-39.));
        let rename_rect=Rect::from_min_size(Pos2::new(rect.right()-87.,rect.bottom()-35.),Vec2::new(73.,26.));
        let bottom_open=Rect::from_min_max(Pos2::new(rect.left(),open_rect.bottom()),Pos2::new(rename_rect.left()-4.,rect.bottom()));
        let open=ui.interact(open_rect,egui::Id::new(("save_card_open",entry.id)),Sense::click());
        let lower=ui.interact(bottom_open,egui::Id::new(("save_card_lower",entry.id)),Sense::click());
        let hovered=open.hovered()||lower.hovered();
        let painter=ui.painter();
        painter.rect_filled(rect.translate(Vec2::new(0.,3.)),10.,Color32::from_black_alpha(6));
        painter.rect_filled(rect,10.,Color32::WHITE);
        painter.rect_stroke(rect,10.,Stroke::new(1.,if hovered{Color32::from_rgb(178,198,240)}else{LINE}));
        let preview=Rect::from_min_size(rect.min+Vec2::splat(14.),Vec2::new(width-28.,preview_height));
        painter.rect_filled(preview,6.,Color32::from_rgb(247,249,252));
        if ui.is_rect_visible(rect){
            if let Some((texture,size))=entry.preview_path.and_then(|path|self.previews.texture(ui.ctx(),path)){
                let fit=(preview.width()/size.x).min(preview.height()/size.y);
                let image=Rect::from_center_size(preview.center(),size*fit);
                painter.image(texture,image,Rect::from_min_max(Pos2::ZERO,Pos2::new(1.,1.)),Color32::WHITE);
            }else{paint_preview_placeholder(painter,preview);}
            let label_y=preview.bottom()+8.;
            painter.text(Pos2::new(preview.left()+preview.width()*0.25,label_y),Align2::CENTER_TOP,"画布",FontId::proportional(10.),MUTED);
            painter.text(Pos2::new(preview.left()+preview.width()*0.75,label_y),Align2::CENTER_TOP,"状态树",FontId::proportional(10.),MUTED);
            let mut job=egui::text::LayoutJob::simple_singleline(entry.name.to_owned(),FontId::proportional(16.),INK);
            job.wrap=egui::text::TextWrapping::truncate_at_width(width-28.);
            let galley=ui.fonts(|fonts|fonts.layout_job(job));
            painter.galley(Pos2::new(rect.left()+14.,preview.bottom()+29.),galley,INK);
            let detail=if entry.updated_label.is_empty(){format!("{} 个状态",entry.node_count)}
                else{format!("{} 个状态 · {}",entry.node_count,entry.updated_label)};
            painter.with_clip_rect(Rect::from_min_max(Pos2::new(rect.left()+14.,rect.bottom()-32.),Pos2::new(rename_rect.left()-8.,rect.bottom()-6.)))
                .text(Pos2::new(rect.left()+14.,rect.bottom()-21.),Align2::LEFT_CENTER,detail,FontId::proportional(11.),MUTED);
        }
        let rename=ui.put(rename_rect,egui::Button::new(RichText::new("重命名").size(12.).color(MUTED)).frame(false));
        #[cfg(test)]{self.card_rects.push((entry.id.into(),open_rect));self.rename_rects.push((entry.id.into(),rename.rect));}
        if open.clicked()||lower.clicked(){actions.push(Action::Open{id:entry.id.into()});}
        open.on_hover_text(entry.name);lower.on_hover_text(entry.name);
        if rename.clicked(){self.rename_dialog(entry.id,entry.name);}
    }

    fn show_name_dialog(&mut self,ctx:&egui::Context,entries:&[SaveEntryView<'_>],busy:bool,actions:&mut Vec<Action>){
        let Some(mut dialog)=self.dialog.take()else{return;};let just_opened=dialog.focus;
        let id=egui::Id::new("save_library_name");
        let composing_event=ctx.input(|input|input.events.iter().any(|event|matches!(event,
            egui::Event::CompositionStart|egui::Event::CompositionUpdate(_)|egui::Event::CompositionEnd(_))));
        if !busy&&ctx.memory(|memory|memory.has_focus(id)){
            ctx.input_mut(|input|repair_name_ime(&mut input.events,&mut dialog.composing));
        }else{dialog.composing=false;}
        let key=|key|ctx.input(|input|input.events.iter().any(|event|matches!(event,
            egui::Event::Key{key:pressed,pressed:true,repeat:false,modifiers,..}if *pressed==key&&!modifiers.command&&!modifiers.ctrl&&!modifiers.alt)));
        let cancel_key=!dialog.composing&&!composing_event&&key(egui::Key::Escape);
        let enter=!just_opened&&!dialog.composing&&!composing_event&&key(egui::Key::Enter);
        let mut open=true;let mut cancel=cancel_key;let mut confirm=false;
        let title=match dialog.kind{NameKind::Create=>"新建存档",NameKind::Rename{..}=>"重命名存档"};
        egui::Window::new(title).id(egui::Id::new("save_library_name_dialog")).open(&mut open)
            .anchor(Align2::CENTER_CENTER,Vec2::ZERO).collapsible(false).resizable(false).default_width(390.).show(ctx,|ui|{
                ui.label("存档名称");ui.add_space(5.);
                let response=ui.add_enabled(!busy,egui::TextEdit::singleline(&mut dialog.text).id(id)
                    .hint_text("为这棵状态树起个名字").desired_width(f32::INFINITY));
                #[cfg(test)]{self.name_input_rect=response.rect;}
                if dialog.focus&&!busy{response.request_focus();dialog.focus=false;}
                let validation=name_error(&dialog.text).or_else(||match &dialog.kind{
                    NameKind::Rename{id}if !entries.iter().any(|entry|entry.id==id)=>Some("这个存档已不在列表中，请关闭后重新选择"),_=>None});
                // Single-line TextEdit releases focus on Enter. If the name
                // is invalid, keep typing in that field instead of leaving
                // the user on a disabled confirmation button.
                if enter&&validation.is_some()&&!busy{response.request_focus();}
                ui.add_space(8.);
                if let Some(error)=validation{ui.label(RichText::new(error).size(12.).color(Color32::from_rgb(177,65,59)));}
                else{ui.label(RichText::new("以后可以随时修改名称。").size(12.).color(MUTED));}
                ui.add_space(15.);ui.with_layout(Layout::right_to_left(Align::Center),|ui|{
                    let label=if matches!(dialog.kind,NameKind::Create){"创建"}else{"保存名称"};
                    let response=ui.add_enabled(!busy&&validation.is_none(),egui::Button::new(RichText::new(label).color(Color32::WHITE))
                        .fill(BLUE).min_size(Vec2::new(92.,32.)));
                    #[cfg(test)]{self.confirm_rect=response.rect;}
                    confirm=response.clicked()||(!busy&&validation.is_none()&&enter);
                    let response=ui.add(egui::Button::new("取消").min_size(Vec2::new(78.,32.)));
                    #[cfg(test)]{self.cancel_rect=response.rect;}
                    cancel|=response.clicked();
                });
            });
        if cancel||!open{return;}
        if confirm{let name=dialog.text.trim().to_owned();actions.push(match dialog.kind{
            NameKind::Create=>Action::Create{name},NameKind::Rename{id}=>Action::Rename{id,name}});}
        else{self.dialog=Some(dialog);}
    }
}

fn name_error(text:&str)->Option<&'static str>{
    let text=text.trim();
    if text.is_empty(){Some("请输入存档名称")}
    else if text.chars().count()>NAME_LIMIT{Some("名称最多可以包含120个字符")}
    else if text.chars().any(char::is_control){Some("名称不能包含换行或控制字符")}
    else{None}
}

fn repair_name_ime(events:&mut[egui::Event],composing:&mut bool){
    for event in events{match event{
        egui::Event::CompositionStart|egui::Event::CompositionUpdate(_)=>*composing=true,
        egui::Event::CompositionEnd(text)=>{if !*composing&&!text.is_empty()&&text!="\r"&&text!="\n"{*event=egui::Event::Text(text.clone());}*composing=false;},_=>{}
    }}
}

fn paint_empty_icon(painter:&egui::Painter,rect:Rect){
    let outline=Color32::from_rgb(188,199,219);let main=Rect::from_min_max(rect.min+Vec2::new(5.,14.),rect.max-Vec2::splat(5.));
    painter.rect_filled(main,8.,Color32::from_rgb(227,234,246));painter.rect_stroke(main,8.,Stroke::new(1.5,outline));
    painter.line_segment([rect.min+Vec2::new(6.,14.),rect.min+Vec2::new(6.,5.)],Stroke::new(1.5,outline));
    painter.line_segment([rect.min+Vec2::new(6.,5.),rect.min+Vec2::new(32.,5.)],Stroke::new(1.5,outline));
    painter.line_segment([rect.min+Vec2::new(32.,5.),rect.min+Vec2::new(41.,14.)],Stroke::new(1.5,outline));
}

fn paint_preview_placeholder(painter:&egui::Painter,rect:Rect){
    let mid=rect.center().x;let canvas=Rect::from_center_size(Pos2::new(rect.left()+rect.width()*0.25,rect.center().y),Vec2::splat(rect.height()*0.73));
    painter.rect_filled(canvas,3.,Color32::WHITE);painter.rect_stroke(canvas,3.,Stroke::new(1.,LINE));
    painter.vline(mid,rect.y_range(),Stroke::new(1.,LINE));
    let center=Pos2::new(rect.left()+rect.width()*0.75,rect.center().y);let gap=rect.height()*0.24;
    for end in [center+Vec2::new(-gap,gap),center+Vec2::new(gap,gap)]{painter.line_segment([center-Vec2::new(0.,gap),center],Stroke::new(1.5,Color32::from_rgb(190,204,229)));
        painter.line_segment([center,end],Stroke::new(1.5,Color32::from_rgb(190,204,229)));painter.circle_filled(end,4.,Color32::from_rgb(190,204,229));}
    painter.circle_filled(center-Vec2::new(0.,gap),4.,Color32::from_rgb(150,178,232));painter.circle_filled(center,4.,Color32::from_rgb(150,178,232));
}

#[derive(Clone,PartialEq,Eq)]struct FileStamp{length:u64,modified:Option<SystemTime>}
struct CachedPreview{stamp:Option<FileStamp>,last_checked:Instant,last_frame:u64,texture:Option<TextureHandle>}
#[derive(Default)]struct PreviewCache{items:HashMap<PathBuf,CachedPreview>,serial:u64}
impl PreviewCache{
    fn retain(&mut self,entries:&[SaveEntryView<'_>]){
        let paths=entries.iter().filter_map(|entry|entry.preview_path).collect::<HashSet<_>>();self.items.retain(|path,_|paths.contains(path.as_path()));
    }
    fn texture(&mut self,ctx:&egui::Context,path:&Path)->Option<(TextureId,Vec2)>{
        let frame=ctx.frame_nr();
        if let Some(cached)=self.items.get_mut(path){cached.last_frame=frame;
            if cached.last_checked.elapsed()<Duration::from_secs(1){return cached.texture.as_ref().map(|texture|(texture.id(),texture.size_vec2()));}}
        let stamp=std::fs::metadata(path).ok().filter(|metadata|metadata.is_file()).map(|metadata|FileStamp{length:metadata.len(),modified:metadata.modified().ok()});
        if let Some(cached)=self.items.get_mut(path){cached.last_checked=Instant::now();if cached.stamp==stamp{
            return cached.texture.as_ref().map(|texture|(texture.id(),texture.size_vec2()));}}
        let image=stamp.as_ref().filter(|stamp|stamp.length<=16*1024*1024).and_then(|_|load_preview(path));
        let texture=image.map(|image|{self.serial=self.serial.wrapping_add(1);ctx.load_texture(format!("save-library-preview-{}",self.serial),image,egui::TextureOptions::LINEAR)});
        if let Some(cached)=self.items.get_mut(path){cached.stamp=stamp;cached.last_checked=Instant::now();cached.last_frame=frame;
            if texture.is_some(){cached.texture=texture;}return cached.texture.as_ref().map(|texture|(texture.id(),texture.size_vec2()));}
        while self.items.len()>=CACHE_TARGET{
            let oldest=self.items.iter().filter(|(_,cached)|cached.last_frame<frame).min_by_key(|(_,cached)|cached.last_frame).map(|(path,_)|path.clone());
            if let Some(path)=oldest{self.items.remove(&path);}else{break;}
        }
        let result=texture.as_ref().map(|texture|(texture.id(),texture.size_vec2()));
        self.items.insert(path.to_owned(),CachedPreview{stamp,last_checked:Instant::now(),last_frame:frame,texture});result
    }
}

fn load_preview(path:&Path)->Option<egui::ColorImage>{
    let mut reader=image::io::Reader::open(path).ok()?.with_guessed_format().ok()?;
    let mut limits=image::io::Limits::default();limits.max_image_width=Some(4096);limits.max_image_height=Some(4096);limits.max_alloc=Some(64*1024*1024);reader.limits(limits);
    let image=reader.decode().ok()?.thumbnail(640,360).into_rgba8();
    if image.width()==0||image.height()==0{return None;}
    Some(egui::ColorImage::from_rgba_unmultiplied([image.width()as usize,image.height()as usize],image.as_raw()))
}

#[cfg(test)]
mod tests{
    use super::*;
    fn entries()->Vec<SaveEntryView<'static>>{vec![SaveEntryView{id:"first",name:"第一棵树",updated_label:"今天",node_count:12,preview_path:None}]}
    fn frame(ctx:&egui::Context,ui:&mut SaveLibraryUi,entries:&[SaveEntryView<'_>],busy:bool,events:Vec<egui::Event>)->Vec<Action>{
        let mut actions=Vec::new();let _=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(1100.,760.))),events,..Default::default()},|ctx|actions=ui.show(ctx,entries,busy,None));actions
    }
    fn pointer(point:Pos2,pressed:bool)->egui::Event{egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()}}
    fn click(ctx:&egui::Context,ui:&mut SaveLibraryUi,entries:&[SaveEntryView<'_>],point:Pos2)->Vec<Action>{
        let mut result=frame(ctx,ui,entries,false,vec![egui::Event::PointerMoved(point),pointer(point,true)]);
        result.extend(frame(ctx,ui,entries,false,vec![pointer(point,false)]));result
    }
    fn key(key:egui::Key)->egui::Event{egui::Event::Key{key,physical_key:None,pressed:true,repeat:false,modifiers:Default::default()}}
    fn release(key:egui::Key)->egui::Event{egui::Event::Key{key,physical_key:None,pressed:false,repeat:false,modifiers:Default::default()}}
    #[test]fn library_offers_one_creation_entry_without_archive_type_buttons_or_labels(){
        let ctx=egui::Context::default();let mut ui=SaveLibraryUi::default();let mut actions=Vec::new();
        let output=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(1100.,760.))),..Default::default()},|ctx|actions=ui.show(ctx,&[],false,None));
        fn text(shape:&egui::epaint::Shape)->String{match shape{egui::epaint::Shape::Text(value)=>value.galley.job.text.clone(),egui::epaint::Shape::Vec(shapes)=>shapes.iter().map(text).collect::<Vec<_>>().join(" "),_=>String::new()}}
        let labels=output.shapes.iter().map(|shape|text(&shape.shape)).collect::<Vec<_>>().join(" ");
        assert!(!labels.contains("修改模式")&&!labels.contains("标准模式"));assert!(actions.is_empty());
        assert!(ui.create_button_rect.is_positive()&&ui.import_button_rect.is_positive());
        let create=ui.create_button_rect.center();assert!(click(&ctx,&mut ui,&[],create).is_empty());
        assert!(matches!(ui.dialog.as_ref().map(|dialog|&dialog.kind),Some(NameKind::Create)));
    }
    #[test]fn empty_library_never_creates_or_opens_a_project_automatically(){
        let ctx=egui::Context::default();let mut ui=SaveLibraryUi::default();for _ in 0..4{assert!(frame(&ctx,&mut ui,&[],false,vec![]).is_empty());}
        assert!(ui.dialog.is_none());assert!(ui.create_button_rect.is_positive());
        assert!(frame(&ctx,&mut ui,&[],true,vec![key(egui::Key::Enter)]).is_empty());
    }
    #[test]fn card_and_rename_button_have_separate_actions(){
        let ctx=egui::Context::default();let mut ui=SaveLibraryUi::default();let entries=entries();frame(&ctx,&mut ui,&entries,false,vec![]);
        let point=ui.card_rects[0].1.center();assert_eq!(click(&ctx,&mut ui,&entries,point),vec![Action::Open{id:"first".into()}]);
        let point=ui.rename_rects[0].1.center();assert!(click(&ctx,&mut ui,&entries,point).is_empty());assert!(ui.dialog.is_some());
        frame(&ctx,&mut ui,&entries,false,vec![]);let point=ui.name_input_rect.center();assert!(point.is_finite());
        frame(&ctx,&mut ui,&entries,false,vec![egui::Event::Key{key:egui::Key::A,physical_key:None,pressed:true,repeat:false,
            modifiers:egui::Modifiers{ctrl:true,command:true,..Default::default()}},egui::Event::Text("  新的名字  ".into())]);
        assert_eq!(frame(&ctx,&mut ui,&entries,false,vec![key(egui::Key::Enter)]),vec![Action::Rename{id:"first".into(),name:"新的名字".into()}]);
    }
    #[test]fn creation_requires_a_name_and_explicit_confirmation_and_escape_cancels(){
        let ctx=egui::Context::default();let mut ui=SaveLibraryUi::default();frame(&ctx,&mut ui,&[],false,vec![]);
        let point=ui.create_button_rect.center();assert!(click(&ctx,&mut ui,&[],point).is_empty());
        frame(&ctx,&mut ui,&[],false,vec![]);
        assert!(frame(&ctx,&mut ui,&[],false,vec![key(egui::Key::Enter)]).is_empty());
        assert!(ctx.memory(|memory|memory.has_focus(egui::Id::new("save_library_name"))));
        assert!(frame(&ctx,&mut ui,&[],false,vec![release(egui::Key::Enter),egui::Event::Text("  我的存档  ".into())]).is_empty());
        assert_eq!(ui.dialog.as_ref().unwrap().text,"  我的存档  ");
        assert_eq!(frame(&ctx,&mut ui,&[],false,vec![key(egui::Key::Enter)]),vec![Action::Create{name:"我的存档".into()}]);
        ui.create_dialog();frame(&ctx,&mut ui,&[],false,vec![]);
        assert!(frame(&ctx,&mut ui,&[],false,vec![key(egui::Key::Escape)]).is_empty());assert!(ui.dialog.is_none());
    }
    #[test]fn busy_or_missing_entry_blocks_confirmation_without_losing_name(){
        let ctx=egui::Context::default();let mut ui=SaveLibraryUi::default();ui.rename_dialog("gone","旧名字");
        frame(&ctx,&mut ui,&[],false,vec![]);assert!(frame(&ctx,&mut ui,&[],false,vec![key(egui::Key::Enter)]).is_empty());
        ui.create_dialog();ui.dialog.as_mut().unwrap().text="保留输入".into();frame(&ctx,&mut ui,&[],true,vec![]);
        assert!(frame(&ctx,&mut ui,&[],true,vec![key(egui::Key::Enter)]).is_empty());assert_eq!(ui.dialog.as_ref().unwrap().text,"保留输入");
        assert_eq!(name_error("\n"),Some("请输入存档名称"));assert!(name_error(&"字".repeat(NAME_LIMIT+1)).is_some());
    }
}
