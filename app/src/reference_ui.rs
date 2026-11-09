//! Native reference preparation. This component never mutates the document or
//! starts I/O itself; every operation is an explicit controller action.
use eframe::egui::{self,Align,Color32,Pos2,Rect,RichText,Sense,Stroke,TextureHandle,Vec2};
use std::sync::Arc;

const INK:Color32=Color32::from_rgb(34,43,62);
const MUTED:Color32=Color32::from_rgb(111,122,143);
const BLUE:Color32=Color32::from_rgb(58,110,242);
pub const DEFAULT_STOP_RMSE:f64=crate::reference_serialize::MIN_STOP_RMSE;

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct PixelCrop{pub x:u32,pub y:u32,pub size:u32}
impl PixelCrop{
    pub fn valid_for(&self,size:[usize;2])->bool{
        self.size>0&&(self.x as usize).checked_add(self.size as usize).is_some_and(|n|n<=size[0])
            &&(self.y as usize).checked_add(self.size as usize).is_some_and(|n|n<=size[1])
    }
    fn centered(size:[usize;2])->Option<Self>{
        let side=size[0].min(size[1]);if side==0||size.iter().any(|&n|n>u32::MAX as usize){return None;}
        Some(Self{x:((size[0]-side)/2)as u32,y:((size[1]-side)/2)as u32,size:side as u32})
    }
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum ReferenceStage{Loading,Crop,Prepare}
#[derive(Clone)]
pub struct ReferenceImageView{pub revision:u64,pub name:String,pub size:[usize;2],pub image:Arc<egui::ColorImage>}
#[derive(Clone)]
pub struct ReferenceResultView{
    pub preview:ReferenceImageView,pub actual_rmse:f64,pub shape_count:usize,
    pub requested_rmse:f64,pub stop_reason:String,
}
#[derive(Clone)]
pub struct ReferenceView{
    pub stage:ReferenceStage,pub source:Option<ReferenceImageView>,pub target:Option<ReferenceImageView>,
    pub result:Option<ReferenceResultView>,pub busy:bool,pub progress:Option<f32>,
    pub status:String,pub error:Option<String>,pub stop_rmse:f64,pub can_import:bool,
}
impl Default for ReferenceView{
    fn default()->Self{Self{stage:ReferenceStage::Loading,source:None,target:None,result:None,busy:false,progress:None,
        status:String::new(),error:None,stop_rmse:DEFAULT_STOP_RMSE,can_import:false}}
}
#[derive(Clone,Debug,PartialEq)]
pub enum ReferenceAction{
    ChooseImage,ConfirmCrop{crop:PixelCrop},Serialize{stop_rmse:f64},Import,
    CancelSerialization,BackToCrop,Close,
}

#[derive(Default)]
struct ImageTexture{key:Option<(u64,[usize;2],usize)>,texture:Option<TextureHandle>}
impl ImageTexture{
    fn get(&mut self,ctx:&egui::Context,image:&ReferenceImageView,name:&str)->egui::TextureId{
        let key=(image.revision,image.size,Arc::as_ptr(&image.image)as usize);
        if self.key!=Some(key){
            if let Some(texture)=&mut self.texture{texture.set((*image.image).clone(),egui::TextureOptions::LINEAR);}
            else{self.texture=Some(ctx.load_texture(name,(*image.image).clone(),egui::TextureOptions::LINEAR));}
            self.key=Some(key);
        }
        self.texture.as_ref().unwrap().id()
    }
}

pub struct ReferenceUi{
    view:Option<ReferenceView>,crop:Option<PixelCrop>,crop_source:Option<u64>,drag_start:Option<Pos2>,
    stop_rmse:f64,source_texture:ImageTexture,target_texture:ImageTexture,result_texture:ImageTexture,
    #[cfg(test)]pub(super) crop_board:Rect,
    #[cfg(test)]pub(super) confirm_rect:Rect,
    #[cfg(test)]pub(super) serialize_rect:Rect,
    #[cfg(test)]pub(super) import_rect:Rect,
}
impl Default for ReferenceUi{
    fn default()->Self{Self{view:None,crop:None,crop_source:None,drag_start:None,stop_rmse:DEFAULT_STOP_RMSE,
        source_texture:Default::default(),target_texture:Default::default(),result_texture:Default::default(),
        #[cfg(test)]crop_board:Rect::NOTHING,#[cfg(test)]confirm_rect:Rect::NOTHING,
        #[cfg(test)]serialize_rect:Rect::NOTHING,#[cfg(test)]import_rect:Rect::NOTHING}}
}
impl ReferenceUi{
    pub fn set_view(&mut self,view:Option<ReferenceView>){
        let source=view.as_ref().and_then(|v|v.source.as_ref()).map(|v|v.revision);
        if self.view.is_none()&&view.is_some(){self.stop_rmse=view.as_ref().unwrap().stop_rmse;}
        if source!=self.crop_source{
            self.crop=view.as_ref().and_then(|v|v.source.as_ref()).and_then(|v|PixelCrop::centered(v.size));
            self.crop_source=source;self.drag_start=None;
            if let Some(view)=&view{self.stop_rmse=view.stop_rmse;}
        }
        if view.is_none(){*self=Self::default();}else{self.view=view;}
    }
    pub fn is_open(&self)->bool{self.view.is_some()}
    pub fn stop_rmse(&self)->Option<f64>{self.view.as_ref().map(|_|self.stop_rmse)}
    pub fn show(&mut self,ctx:&egui::Context)->Vec<ReferenceAction>{
        let Some(view)=self.view.clone()else{return Vec::new();};let mut actions=Vec::new();
        if ctx.input(|i|i.key_pressed(egui::Key::Escape)){actions.push(ReferenceAction::Close);return actions;}
        egui::TopBottomPanel::top("reference_page_header").frame(egui::Frame::none().fill(Color32::WHITE).inner_margin(18.)).show(ctx,|ui|{
            ui.horizontal(|ui|{
                if ui.button("← 返回画布").clicked(){actions.push(ReferenceAction::Close);}
                ui.add_space(12.);ui.label(RichText::new(if view.stage==ReferenceStage::Crop{"框选参考图像"}else{"参考制作"}).size(20.).strong().color(INK));
                ui.with_layout(egui::Layout::right_to_left(Align::Center),|ui|{
                    if ui.add_enabled(!view.busy,egui::Button::new("重新选择图像")).clicked(){actions.push(ReferenceAction::ChooseImage);}
                });
            });
        });
        egui::CentralPanel::default().frame(egui::Frame::none().fill(Color32::from_rgb(242,245,250)).inner_margin(20.)).show(ctx,|ui|{
            if let Some(error)=&view.error{ui.colored_label(Color32::from_rgb(177,65,59),error);ui.add_space(8.);}
            match view.stage{
                ReferenceStage::Loading=>{ui.vertical_centered(|ui|{ui.add_space(80.);ui.spinner();ui.label(if view.status.is_empty(){"正在读取参考图像…"}else{&view.status});});},
                ReferenceStage::Crop=>self.crop_page(ui,&view,&mut actions),
                ReferenceStage::Prepare=>self.prepare_page(ui,&view,&mut actions),
            }
        });
        if view.busy{ctx.request_repaint_after(std::time::Duration::from_millis(80));}
        actions
    }

    fn crop_page(&mut self,ui:&mut egui::Ui,view:&ReferenceView,actions:&mut Vec<ReferenceAction>){
        let Some(source)=&view.source else{ui.label("请先选择参考图像。");return;};
        ui.horizontal(|ui|{
            ui.label(RichText::new(format!("{}  ·  {} × {}",source.name,source.size[0],source.size[1])).color(INK));
            ui.label(RichText::new("拖动框选正方形参考范围").size(12.).color(MUTED));
        });ui.add_space(10.);
        let available=Vec2::new(ui.available_width(),(ui.available_height()-62.).max(100.));
        let (viewport,_)=ui.allocate_exact_size(available,Sense::hover());
        let scale=(viewport.width()/source.size[0].max(1)as f32).min(viewport.height()/source.size[1].max(1)as f32);
        let board=Rect::from_center_size(viewport.center(),Vec2::new(source.size[0]as f32,source.size[1]as f32)*scale);
        #[cfg(test)]{self.crop_board=board;}
        let response=ui.interact(board,egui::Id::new("reference_crop_image"),if view.busy{Sense::hover()}else{Sense::click_and_drag()});
        ui.painter().rect_filled(board,0.,Color32::WHITE);
        let texture=self.source_texture.get(ui.ctx(),source,"reference-source");
        ui.painter().image(texture,board,Rect::from_min_max(Pos2::ZERO,Pos2::new(1.,1.)),Color32::WHITE);
        let pixel=|p:Pos2|Pos2::new(((p.x-board.left())/scale).clamp(0.,source.size[0]as f32),((p.y-board.top())/scale).clamp(0.,source.size[1]as f32));
        if response.drag_started(){self.drag_start=ui.input(|i|i.pointer.press_origin()).filter(|p|board.contains(*p)).map(pixel);}
        if response.dragged()||response.drag_stopped(){
            if let(Some(start),Some(now))=(self.drag_start,ui.input(|i|i.pointer.interact_pos())){
                self.crop=drag_crop(start,pixel(now),source.size);
            }
        }
        if response.drag_stopped(){self.drag_start=None;}
        if let Some(crop)=self.crop.filter(|c|c.valid_for(source.size)){
            let selected=Rect::from_min_size(board.min+Vec2::new(crop.x as f32,crop.y as f32)*scale,Vec2::splat(crop.size as f32*scale));
            let shade=Color32::from_black_alpha(115);
            for r in [Rect::from_min_max(board.min,Pos2::new(board.right(),selected.top())),
                Rect::from_min_max(Pos2::new(board.left(),selected.bottom()),board.max),
                Rect::from_min_max(Pos2::new(board.left(),selected.top()),Pos2::new(selected.left(),selected.bottom())),
                Rect::from_min_max(Pos2::new(selected.right(),selected.top()),Pos2::new(board.right(),selected.bottom()))]{
                if r.is_positive(){ui.painter().rect_filled(r,0.,shade);}
            }
            ui.painter().rect_stroke(selected,0.,Stroke::new(2.,BLUE));
        }
        ui.add_space(10.);ui.horizontal(|ui|{
            if let Some(crop)=self.crop{ui.label(RichText::new(format!("选区：{} × {} 像素",crop.size,crop.size)).size(12.).color(MUTED));}
            ui.with_layout(egui::Layout::right_to_left(Align::Center),|ui|{
                let enabled=!view.busy&&self.crop.is_some_and(|c|c.valid_for(source.size));
                let button=ui.add_enabled(enabled,primary("确认截取，进入参考制作"));
                #[cfg(test)]{self.confirm_rect=button.rect;}
                if button.clicked(){actions.push(ReferenceAction::ConfirmCrop{crop:self.crop.unwrap()});}
            });
        });
    }

    fn prepare_page(&mut self,ui:&mut egui::Ui,view:&ReferenceView,actions:&mut Vec<ReferenceAction>){
        ui.horizontal_wrapped(|ui|{
            ui.label(RichText::new("截止 RMSE").strong().color(INK));
            ui.add_enabled(!view.busy,egui::DragValue::new(&mut self.stop_rmse).speed(0.25)
                .clamp_range(crate::reference_serialize::MIN_STOP_RMSE..=crate::reference_serialize::MAX_STOP_RMSE).max_decimals(3))
                .on_hover_text("允许范围：12.75–255。12.75 为训练集标准，数值越大要求越宽松。");
            let valid=crate::reference_serialize::validate_stop_rmse(self.stop_rmse).is_ok();
            let serialize=ui.add_enabled(!view.busy&&valid&&view.target.is_some(),primary("序列化"));
            #[cfg(test)]{self.serialize_rect=serialize.rect;}
            if serialize.clicked(){actions.push(ReferenceAction::Serialize{stop_rmse:self.stop_rmse});}
            if view.busy&&ui.button("取消序列化").clicked(){actions.push(ReferenceAction::CancelSerialization);}
            let matches=view.result.as_ref().is_some_and(|r|(r.requested_rmse-self.stop_rmse).abs()<1e-9);
            let import=ui.add_enabled(!view.busy&&valid&&view.can_import&&matches,primary("导入参考"));
            #[cfg(test)]{self.import_rect=import.rect;}
            if import.clicked(){actions.push(ReferenceAction::Import);}
            if ui.add_enabled(!view.busy,egui::Button::new("重新框选")).clicked(){actions.push(ReferenceAction::BackToCrop);}
        });
        ui.label(RichText::new("阈值不得低于训练集标准 12.75，数值越大要求越宽松。全图阶段检查整图，粗、细阶段检查本格，并保留每格图元数上限。结果需手动导入。").size(12.).color(MUTED));
        if view.busy{ui.horizontal(|ui|{ui.spinner();ui.label(&view.status);});}
        else if !view.status.is_empty(){ui.label(RichText::new(&view.status).color(MUTED));}
        if let Some(progress)=view.progress.filter(|p|p.is_finite()){
            ui.add(egui::ProgressBar::new(progress.clamp(0.,1.)).show_percentage());
        }
        if let Some(result)=&view.result{
            ui.horizontal_wrapped(|ui|{
                ui.label(RichText::new(format!("全图 RMSE：{:.3}  ·  图元数：{}",result.actual_rmse,result.shape_count)).color(INK));
                if !result.stop_reason.is_empty(){ui.label(RichText::new(&result.stop_reason).color(MUTED));}
            });
            if (result.requested_rmse-self.stop_rmse).abs()>=1e-9{
                ui.colored_label(Color32::from_rgb(155,111,32),"截止 RMSE 已修改，请重新序列化后再导入。");
            }
        }
        ui.add_space(12.);
        let side=((ui.available_width()-20.)*0.5).min((ui.available_height()-28.).max(60.));
        ui.columns(2,|columns|{
            columns[0].label(RichText::new("截取的参考图像").strong().color(INK));
            if let Some(target)=&view.target{draw_image(&mut columns[0],&mut self.target_texture,target,"reference-target",side);}
            columns[1].label(RichText::new("序列化结果预览").strong().color(INK));
            if let Some(result)=&view.result{draw_image(&mut columns[1],&mut self.result_texture,&result.preview,"reference-result",side);}
            else{let(rect,_)=columns[1].allocate_exact_size(Vec2::splat(side),Sense::hover());columns[1].painter().rect_filled(rect,0.,Color32::WHITE);
                columns[1].painter().text(rect.center(),egui::Align2::CENTER_CENTER,"尚未序列化",egui::FontId::proportional(14.),MUTED);}
        });
    }
}

fn primary(text:&str)->egui::Button<'_>{
    egui::Button::new(RichText::new(text).color(Color32::WHITE)).fill(BLUE).min_size(Vec2::new(100.,32.))
}
fn draw_image(ui:&mut egui::Ui,cache:&mut ImageTexture,image:&ReferenceImageView,name:&str,side:f32){
    let(rect,_)=ui.allocate_exact_size(Vec2::splat(side),Sense::hover());ui.painter().rect_filled(rect,0.,Color32::WHITE);
    let texture=cache.get(ui.ctx(),image,name);ui.painter().image(texture,rect,Rect::from_min_max(Pos2::ZERO,Pos2::new(1.,1.)),Color32::WHITE);
}
fn drag_crop(start:Pos2,now:Pos2,size:[usize;2])->Option<PixelCrop>{
    if size[0]==0||size[1]==0||size.iter().any(|&n|n>u32::MAX as usize){return None;}
    let sx=start.x.floor().clamp(0.,size[0].saturating_sub(1)as f32);
    let sy=start.y.floor().clamp(0.,size[1].saturating_sub(1)as f32);
    let left=now.x<sx;let up=now.y<sy;
    let max_x=if left{sx}else{size[0]as f32-sx};let max_y=if up{sy}else{size[1]as f32-sy};
    let side=(now.x-sx).abs().max((now.y-sy).abs()).ceil().min(max_x).min(max_y).max(1.)as u32;
    let crop=PixelCrop{x:if left{sx as u32-side}else{sx as u32},y:if up{sy as u32-side}else{sy as u32},size:side};
    crop.valid_for(size).then_some(crop)
}

#[cfg(test)]
mod tests{
    use super::*;
    fn image(revision:u64,size:[usize;2])->ReferenceImageView{
        ReferenceImageView{revision,name:"参考图像.png".into(),size,
            image:Arc::new(egui::ColorImage::new(size,Color32::from_rgb(210,180,150)))}
    }
    fn prepared()->ReferenceView{
        let target=image(2,[64,64]);let preview=image(3,[64,64]);
        ReferenceView{stage:ReferenceStage::Prepare,source:Some(image(1,[128,64])),target:Some(target),
            result:Some(ReferenceResultView{preview,actual_rmse:20.,shape_count:86,requested_rmse:DEFAULT_STOP_RMSE,stop_reason:"达到本轮图元上限".into()}),
            can_import:true,..Default::default()}
    }
    fn frame(ctx:&egui::Context,state:&mut ReferenceUi,events:Vec<egui::Event>)->Vec<ReferenceAction>{
        let mut actions=Vec::new();let _=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(1100.,720.))),events,..Default::default()},|ctx|{
            actions=state.show(ctx);
        });actions
    }
    fn click(ctx:&egui::Context,state:&mut ReferenceUi,point:Pos2)->Vec<ReferenceAction>{
        let mut actions=Vec::new();for pressed in [true,false]{actions.extend(frame(ctx,state,vec![egui::Event::PointerMoved(point),
            egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()}]));}actions
    }
    #[test]
    fn square_crops_use_original_pixel_bounds_and_accept_small_regions(){
        assert_eq!(PixelCrop::centered([1600,900]),Some(PixelCrop{x:350,y:0,size:900}));
        for(start,end)in[(Pos2::new(500.,250.),Pos2::new(0.,0.)),(Pos2::new(10.,10.),Pos2::new(900.,900.)),
            (Pos2::new(1.,1.),Pos2::new(0.,0.)),(Pos2::new(0.,0.),Pos2::new(0.2,0.2))]{
            let crop=drag_crop(start,end,[640,480]).unwrap();assert!(crop.valid_for([640,480]));assert!(crop.size>0);
        }
        assert!(PixelCrop{x:0,y:0,size:1}.valid_for([1,1]));
        assert!(!PixelCrop{x:u32::MAX,y:0,size:10}.valid_for([640,480]));
        assert!(!PixelCrop{x:0,y:0,size:0}.valid_for([640,480]));
    }
    #[test]
    fn crop_confirmation_only_emits_crop_and_never_starts_serialization_or_import(){
        let ctx=egui::Context::default();let mut state=ReferenceUi::default();
        state.set_view(Some(ReferenceView{stage:ReferenceStage::Crop,source:Some(image(1,[128,64])),..Default::default()}));
        for _ in 0..3{assert!(frame(&ctx,&mut state,vec![]).is_empty());}
        let point=state.confirm_rect.center();let actions=click(&ctx,&mut state,point);
        assert_eq!(actions,vec![ReferenceAction::ConfirmCrop{crop:PixelCrop{x:32,y:0,size:64}}]);
        assert_eq!(state.view.as_ref().unwrap().stage,ReferenceStage::Crop,"controller owns the stage transition");
    }
    #[test]
    fn completed_preview_never_imports_automatically_and_rmse_changes_require_reserialization(){
        let ctx=egui::Context::default();let mut state=ReferenceUi::default();state.set_view(Some(prepared()));
        for _ in 0..4{assert!(frame(&ctx,&mut state,vec![]).is_empty());}
        // A valid result may be accepted even if its threshold was not reached.
        let point=state.import_rect.center();assert_eq!(click(&ctx,&mut state,point),vec![ReferenceAction::Import]);
        state.stop_rmse=20.;for _ in 0..2{assert!(frame(&ctx,&mut state,vec![]).is_empty());}
        let point=state.import_rect.center();assert!(click(&ctx,&mut state,point).is_empty());
        let point=state.serialize_rect.center();assert_eq!(click(&ctx,&mut state,point),vec![ReferenceAction::Serialize{stop_rmse:20.}]);
        assert!(state.view.as_ref().unwrap().result.is_some(),"the old preview can remain visible until the controller replaces it");
    }
    #[test]
    fn busy_or_incomplete_reference_cannot_import_and_escape_returns_only_close(){
        let ctx=egui::Context::default();let mut state=ReferenceUi::default();let mut view=prepared();view.busy=true;state.set_view(Some(view));
        for _ in 0..3{assert!(frame(&ctx,&mut state,vec![]).is_empty());}
        let point=state.import_rect.center();assert!(click(&ctx,&mut state,point).is_empty());
        let actions=frame(&ctx,&mut state,vec![egui::Event::Key{key:egui::Key::Escape,physical_key:None,pressed:true,repeat:false,modifiers:Default::default()}]);
        assert_eq!(actions,vec![ReferenceAction::Close]);
        state.set_view(None);assert!(!state.is_open());assert!(state.source_texture.texture.is_none());
    }
}
