use super::{file_dialogs::{self,FileKind},UiAction,UiModel};
use eframe::egui::{self,Align2,Color32,Rect,RichText,Vec2};
use std::path::PathBuf;

enum Modal { Png { resolution:u32, custom:bool }, Import(PathBuf), Error(String) }

fn requires_import_confirmation(model:&UiModel<'_>)->bool{
    model.nodes.len()>1||model.running||!model.primitives.is_empty()||!model.prompt.is_empty()
        ||model.document.nodes.first().is_none_or(|root|model.settings!=&root.settings)
}

pub(super) struct FileMenu {
    project_path:Option<PathBuf>,
    managed:bool,
    busy:bool,
    modal:Option<Modal>,
    #[cfg(test)]
    pub(super) dialog_results:std::collections::VecDeque<Result<Option<PathBuf>,String>>,
    #[cfg(test)]
    pub(super) primary_rect:Rect,
    #[cfg(test)]
    pub(super) cancel_rect:Rect,
    #[cfg(test)]
    pub(super) stop_rect:Rect,
}

impl Default for FileMenu {
    fn default()->Self{Self{project_path:None,managed:false,busy:false,modal:None,
        #[cfg(test)] dialog_results:Default::default(),
        #[cfg(test)] primary_rect:Rect::NOTHING,
        #[cfg(test)] cancel_rect:Rect::NOTHING,
        #[cfg(test)] stop_rect:Rect::NOTHING,
    }}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::{Document,Primitive,OperationSource,ShapeKind},region_state::DrawingRegion,ui::NodeView};
    use egui::{Pos2,Event,Key,Modifiers,PointerButton};

    fn document()->Document{let mut d=Document::new();d.commit_canvas("测试画布",OperationSource::User,
        vec![Primitive::new(ShapeKind::Rectangle,30.,30.,20.,20.,[20,40,60])]).unwrap();d}
    fn key(key:Key,ctrl:bool,shift:bool)->Event{Event::Key{key,physical_key:None,pressed:true,repeat:false,
        modifiers:Modifiers{ctrl,command:ctrl,shift,..Default::default()}}}
    fn frame(ctx:&egui::Context,files:&mut FileMenu,doc:&Document,events:Vec<Event>)->(Vec<UiAction>,egui::FullOutput){
        frame_running(ctx,files,doc,events,true)
    }
    fn frame_running(ctx:&egui::Context,files:&mut FileMenu,doc:&Document,mut events:Vec<Event>,running:bool)->(Vec<UiAction>,egui::FullOutput){
        // Each test shortcut is a complete physical key tap; holding S/O
        // across frames intentionally produces repeat events in egui.
        let releases=events.iter().filter_map(|event|match event{Event::Key{key,physical_key,pressed:true,modifiers,..}=>Some(Event::Key{key:*key,physical_key:*physical_key,pressed:false,repeat:false,modifiers:*modifiers}),_=>None}).collect::<Vec<_>>();
        events.extend(releases);
        let nodes=doc.nodes.iter().map(|n|NodeView{id:n.id as u64,parent:n.parent.map(|p|p as u64),title:n.title.clone(),source:"用户".into(),primitive_count:n.primitive_count}).collect::<Vec<_>>();
        let region=DrawingRegion::default();let model=UiModel{document:doc,drawing_region:&region,primitives:&doc.primitives,stage_by_id:None,prompt:&doc.prompt,
            settings:&doc.settings,current_node:doc.current as u64,nodes:&nodes,running,loading:false,compiling:false,stopping:false,
            model_loaded:true,model_status:"已加载",model_path:"",tokenizer_path:"",error:None,status:"正在导出"};
        let modifiers=events.iter().rev().find_map(|e|if let Event::Key{modifiers,..}=e{Some(*modifiers)}else{None}).unwrap_or_default();
        let mut actions=Vec::new();let output=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(900.,700.))),events,modifiers,..Default::default()},|ctx|{
            if files.busy{actions=files.show_busy(ctx,&model);}
            else if files.has_modal(){actions=files.show_modal(ctx,true);}
            else{egui::CentralPanel::default().show(ctx,|ui|files.menu(ui,&model,&mut actions));files.shortcuts(ctx,&model,&mut actions);}
        });(actions,output)
    }
    fn click(ctx:&egui::Context,files:&mut FileMenu,doc:&Document,point:Pos2)->Vec<UiAction>{
        let mut all=Vec::new();for pressed in [true,false]{all.extend(frame(ctx,files,doc,vec![Event::PointerMoved(point),Event::PointerButton{pos:point,button:PointerButton::Primary,pressed,modifiers:Default::default()}]).0);}all
    }
    fn label(output:&egui::FullOutput,text:&str)->Pos2{
        fn find(shape:&egui::epaint::Shape,text:&str)->Option<Pos2>{match shape{egui::epaint::Shape::Text(t)if t.galley.job.text==text=>Some(t.pos+t.galley.size()*0.5),
            egui::epaint::Shape::Vec(shapes)=>shapes.iter().find_map(|s|find(s,text)),_=>None}}
        output.shapes.iter().find_map(|s|find(&s.shape,text)).unwrap_or_else(||panic!("missing native UI label {text}"))
    }

    #[test]
    fn control_s_reuses_current_path_and_save_as_waits_for_successful_path_sync(){
        let ctx=egui::Context::default();let mut files=FileMenu::default();let doc=document();let old=PathBuf::from("作品甲.pproj");let new=PathBuf::from("作品乙.pproj");files.set_project_path(Some(old.clone()));
        assert!(matches!(frame(&ctx,&mut files,&doc,vec![key(Key::S,true,false)]).0.as_slice(),[UiAction::SaveProject(path)]if path==&old));
        files.dialog_results.push_back(Ok(Some(new.clone())));
        assert!(matches!(frame(&ctx,&mut files,&doc,vec![key(Key::S,true,true)]).0.as_slice(),[UiAction::SaveProject(path)]if path==&new));
        assert!(matches!(frame(&ctx,&mut files,&doc,vec![key(Key::S,true,false)]).0.as_slice(),[UiAction::SaveProject(path)]if path==&old));
        files.set_project_path(Some(new.clone()));
        assert!(matches!(frame(&ctx,&mut files,&doc,vec![key(Key::S,true,false)]).0.as_slice(),[UiAction::SaveProject(path)]if path==&new));
    }

    #[test]
    fn import_native_cancel_and_replacement_cancel_never_stop_or_modify_the_current_tree(){
        let ctx=egui::Context::default();let mut files=FileMenu::default();let doc=document();let count=doc.nodes.len();
        files.dialog_results.push_back(Ok(None));assert!(frame(&ctx,&mut files,&doc,vec![key(Key::O,true,false)]).0.is_empty());assert!(!files.has_modal());
        files.dialog_results.push_back(Ok(Some(PathBuf::from("导入项目.pproj"))));
        assert!(frame(&ctx,&mut files,&doc,vec![key(Key::O,true,false)]).0.is_empty());assert!(files.has_modal());
        assert!(frame(&ctx,&mut files,&doc,vec![key(Key::Escape,false,false)]).0.is_empty());assert!(!files.has_modal());assert_eq!(doc.nodes.len(),count);
    }

    #[test]
    fn empty_root_with_prompt_or_settings_drafts_requires_confirmation_but_default_blank_imports_directly(){
        for draft in 0..3{
            let ctx=egui::Context::default();let mut files=FileMenu::default();let mut doc=Document::new();
            if draft==1{doc.prompt="尚未绘图的提示词".into();}
            if draft==2{doc.settings.cfg+=0.5;}
            let path=PathBuf::from("新的状态树.pproj");files.dialog_results.push_back(Ok(Some(path.clone())));
            let actions=frame_running(&ctx,&mut files,&doc,vec![key(Key::O,true,false)],false).0;
            if draft==0{assert!(matches!(actions.as_slice(),[UiAction::ImportProject(selected)]if selected==&path));assert!(!files.has_modal());}
            else{assert!(actions.is_empty(),"a draft must not import before confirmation");assert!(files.has_modal());}
            assert_eq!(doc.nodes.len(),1);
        }
    }

    #[test]
    fn import_is_dispatched_only_after_the_real_confirmation_button(){
        let ctx=egui::Context::default();let mut files=FileMenu::default();let doc=document();let path=PathBuf::from("目录/作品.pproj");files.dialog_results.push_back(Ok(Some(path.clone())));
        frame(&ctx,&mut files,&doc,vec![key(Key::O,true,false)]);for _ in 0..3{frame(&ctx,&mut files,&doc,vec![]);}
        let point=files.primary_rect.center();let actions=click(&ctx,&mut files,&doc,point);
        assert!(matches!(actions.as_slice(),[UiAction::ImportProject(selected)]if selected==&path));assert!(!files.has_modal());
    }

    #[test]
    fn png_resolution_then_native_save_cancel_or_choice_dispatches_only_an_export(){
        let ctx=egui::Context::default();let mut files=FileMenu::default();let doc=document();files.modal=Some(Modal::Png{resolution:1024,custom:false});
        for _ in 0..3{frame(&ctx,&mut files,&doc,vec![]);}let (_,output)=frame(&ctx,&mut files,&doc,vec![]);let point=label(&output,"4096");assert!(click(&ctx,&mut files,&doc,point).is_empty());
        files.dialog_results.push_back(Ok(Some(PathBuf::from("示例.png"))));let point=files.primary_rect.center();
        assert!(matches!(click(&ctx,&mut files,&doc,point).as_slice(),[UiAction::ExportPng{resolution:4096,..}]));
        files.modal=Some(Modal::Png{resolution:8192,custom:true});for _ in 0..3{frame(&ctx,&mut files,&doc,vec![]);}
        files.dialog_results.push_back(Ok(None));let point=files.primary_rect.center();assert!(click(&ctx,&mut files,&doc,point).is_empty());assert!(!files.has_modal());
    }

    #[test]
    fn actual_file_menu_svg_and_cancel_routes_do_not_stop_generation(){
        let ctx=egui::Context::default();let mut files=FileMenu::default();let doc=document();
        let (_,output)=frame(&ctx,&mut files,&doc,vec![]);let point=label(&output,"文件");click(&ctx,&mut files,&doc,point);
        let (_,output)=frame(&ctx,&mut files,&doc,vec![]);let point=label(&output,"导出 SVG…");files.dialog_results.push_back(Ok(Some(PathBuf::from("矢量.svg"))));
        assert!(matches!(click(&ctx,&mut files,&doc,point).as_slice(),[UiAction::ExportSvg(_)]));
    }

    #[test]
    fn file_busy_blocks_shortcuts_but_escape_and_the_stop_button_still_work(){
        let ctx=egui::Context::default();let mut files=FileMenu::default();let doc=document();files.set_busy(true);
        assert!(frame(&ctx,&mut files,&doc,vec![key(Key::S,true,false),key(Key::O,true,false),key(Key::Delete,false,false)]).0.is_empty());
        assert!(matches!(frame(&ctx,&mut files,&doc,vec![key(Key::Escape,false,false)]).0.as_slice(),[UiAction::Stop]));
        frame(&ctx,&mut files,&doc,vec![]);let point=files.stop_rect.center();assert!(matches!(click(&ctx,&mut files,&doc,point).as_slice(),[UiAction::Stop]));assert!(files.is_busy());
    }
}

impl FileMenu {
    fn action_pending(actions:&[UiAction])->bool{actions.iter().any(|action|matches!(action,
        UiAction::SaveProject(_)|UiAction::ImportProject(_)|UiAction::ExportPng{..}|UiAction::ExportSvg(_)))}
    pub(super) fn set_managed(&mut self,value:bool){self.managed=value;}
    pub(super) fn set_project_path(&mut self,path:Option<PathBuf>){self.project_path=path;}
    pub(super) fn set_busy(&mut self,busy:bool){self.busy=busy;if busy{self.modal=None;}}
    pub(super) fn is_busy(&self)->bool{self.busy}
    pub(super) fn has_modal(&self)->bool{self.modal.is_some()}

    fn choose(&mut self,kind:FileKind,save:bool)->Result<Option<PathBuf>,String>{
        #[cfg(test)]
        {let _=(kind,save);self.dialog_results.pop_front().expect("provide a fake file dialog result; tests must never open native dialogs")}
        #[cfg(not(test))]
        {if save{file_dialogs::choose_save(kind,self.project_path.as_deref())}else{file_dialogs::choose_import(self.project_path.as_deref())}}
    }
    fn save(&mut self,save_as:bool,actions:&mut Vec<UiAction>){
        if self.busy||self.has_modal()||Self::action_pending(actions){return;}
        if !save_as {if let Some(path)=&self.project_path {actions.push(UiAction::SaveProject(path.clone()));return;}}
        match self.choose(FileKind::Project,true){Ok(Some(path))=>actions.push(UiAction::SaveProject(path)),Ok(None)=>{},Err(error)=>self.modal=Some(Modal::Error(error))}
    }
    fn import(&mut self,replace:bool,actions:&mut Vec<UiAction>){
        if self.busy||self.has_modal()||Self::action_pending(actions){return;}
        match self.choose(FileKind::Project,false){
            Ok(Some(path))=>if replace{self.modal=Some(Modal::Import(path));}else{actions.push(UiAction::ImportProject(path));},
            Ok(None)=>{},Err(error)=>self.modal=Some(Modal::Error(error)),
        }
    }
    fn export_svg(&mut self,actions:&mut Vec<UiAction>){
        if self.busy||self.has_modal()||Self::action_pending(actions){return;}
        match self.choose(FileKind::Svg,true){Ok(Some(path))=>actions.push(UiAction::ExportSvg(path)),Ok(None)=>{},Err(error)=>self.modal=Some(Modal::Error(error))}
    }

    pub(super) fn shortcuts(&mut self,ctx:&egui::Context,model:&UiModel<'_>,actions:&mut Vec<UiAction>){
        if self.busy||self.has_modal()||Self::action_pending(actions)||model.loading||model.compiling||model.stopping{return;}
        let shortcut=ctx.input(|input|input.events.iter().find_map(|event|match event{
            egui::Event::Key{key,pressed:true,repeat:false,modifiers,..} if (modifiers.command||modifiers.ctrl)&&!modifiers.alt=>
                match key{egui::Key::S=>Some((true,modifiers.shift)),egui::Key::O=>Some((false,false)),_=>None},_=>None}));
        if let Some((save,save_as))=shortcut{
            ctx.input_mut(|input|input.events.retain(|event|!matches!(event,egui::Event::Key{key:egui::Key::S|egui::Key::O,modifiers,..}if modifiers.command||modifiers.ctrl)));
            if save{self.save(save_as,actions);}else{self.import(!self.managed&&requires_import_confirmation(model),actions);}
        }
    }

    pub(super) fn menu(&mut self,ui:&mut egui::Ui,model:&UiModel<'_>,actions:&mut Vec<UiAction>){
        ui.add_enabled_ui(!self.busy&&!self.has_modal()&&!model.loading&&!model.compiling&&!model.stopping,|ui|{
            ui.menu_button("文件",|ui|{
                let name=self.project_path.as_ref().and_then(|p|p.file_name()).map(|n|n.to_string_lossy().into_owned()).unwrap_or_else(||"未保存项目".into());
                ui.label(RichText::new(name).small().color(Color32::GRAY));ui.separator();
                if ui.button("保存状态树    Ctrl+S").clicked(){ui.close_menu();self.save(false,actions);}
                if ui.button("另存为…").clicked(){ui.close_menu();self.save(true,actions);}
                if ui.button("导入状态树…    Ctrl+O").clicked(){ui.close_menu();self.import(!self.managed&&requires_import_confirmation(model),actions);}
                ui.separator();
                if ui.button("导出 PNG…").clicked(){ui.close_menu();self.modal=Some(Modal::Png{resolution:1024,custom:false});}
                if ui.button("导出 SVG…").clicked(){ui.close_menu();self.export_svg(actions);}
            });
        });
    }

    pub(super) fn confirm_same_frame_import_drafts(&mut self,model:&UiModel<'_>,prompt:&str,settings:&crate::document::Settings,actions:&mut Vec<UiAction>){
        if self.managed{return;}
        let has_draft=requires_import_confirmation(model)||!prompt.is_empty()
            ||model.document.nodes.first().is_none_or(|root|settings!=&root.settings);
        if !has_draft{return;}
        if let Some(index)=actions.iter().position(|action|matches!(action,UiAction::ImportProject(_))){
            if let UiAction::ImportProject(path)=actions.remove(index){self.modal=Some(Modal::Import(path));}
        }
    }

    pub(super) fn show_busy(&mut self,ctx:&egui::Context,model:&UiModel<'_>)->Vec<UiAction>{
        let mut actions=Vec::new();
        egui::CentralPanel::default().show(ctx,|_|{});
        let can_stop=model.running&&!model.stopping;
        egui::Window::new("文件操作").id(egui::Id::new("file_busy_modal")).anchor(Align2::CENTER_CENTER,Vec2::ZERO)
            .collapsible(false).resizable(false).show(ctx,|ui|{
                ui.set_min_width(330.);ui.horizontal(|ui|{super::paint_busy_indicator(ui);ui.label("正在处理文件，请稍候…");});
                if model.status.contains("保存")||model.status.contains("导入")||model.status.contains("导出"){ui.label(model.status);}
                if model.running {
                    let response=ui.add_enabled(can_stop,egui::Button::new(if model.stopping{"正在停止生成…"}else{"停止生成（Esc）"}));
                    #[cfg(test)]{self.stop_rect=response.rect;}
                    if response.clicked(){actions.push(UiAction::Stop);}
                }
            });
        if can_stop&&ctx.input(|i|i.key_pressed(egui::Key::Escape))&&!actions.iter().any(|a|matches!(a,UiAction::Stop)){actions.push(UiAction::Stop);}
        ctx.request_repaint_after(std::time::Duration::from_millis(80));
        actions
    }

    pub(super) fn show_modal(&mut self,ctx:&egui::Context,clear_background:bool)->Vec<UiAction>{
        let mut actions=Vec::new();let Some(mut modal)=self.modal.take()else{return actions;};
        if clear_background {egui::CentralPanel::default().show(ctx,|_|{});}
        let mut accept=false;let mut cancel=ctx.input(|i|i.key_pressed(egui::Key::Escape));
        let title=match &modal{Modal::Png{..}=>"导出 PNG",Modal::Import(_)=>"导入状态树",Modal::Error(_)=>"文件操作"};
        egui::Window::new(title).id(egui::Id::new("file_choice_modal")).anchor(Align2::CENTER_CENTER,Vec2::ZERO)
            .collapsible(false).resizable(false).show(ctx,|ui|{
                ui.set_min_width(360.);
                match &mut modal {
                    Modal::Png{resolution,custom}=>{
                        ui.label("输出分辨率");ui.horizontal(|ui|{for size in [512,1024,2048,4096]{
                            if ui.selectable_label(!*custom&&*resolution==size,format!("{size}")).clicked(){*resolution=size;*custom=false;}
                        }});
                        ui.horizontal(|ui|{ui.selectable_value(custom,true,"自定义");ui.add_enabled_ui(*custom,|ui|{ui.add(egui::DragValue::new(resolution).clamp_range(1..=8192).suffix(" px"));});});
                        ui.label(RichText::new(format!("{} × {} 像素",*resolution,*resolution)).small());
                    },
                    Modal::Import(path)=>{
                        ui.label("导入将替换当前状态树。");
                        if let Some(name)=path.file_name(){ui.label(name.to_string_lossy().into_owned());}
                        ui.label("如需保留当前作品，请取消后先保存。");
                    },
                    Modal::Error(message)=>{ui.colored_label(Color32::from_rgb(185,55,55),message.as_str());},
                }
                ui.add_space(10.);ui.horizontal(|ui|{
                    let caption=match modal{Modal::Png{..}=>"选择保存位置…",Modal::Import(_)=>"导入替换",Modal::Error(_)=>"确定"};
                    let primary=ui.button(caption);
                    #[cfg(test)]{self.primary_rect=primary.rect;}
                    accept=primary.clicked();
                    if !matches!(modal,Modal::Error(_)){
                        let response=ui.button("取消");
                        #[cfg(test)]{self.cancel_rect=response.rect;}
                        cancel|=response.clicked();
                    }
                });
            });
        if cancel{return actions;}
        if accept {
            match modal {
                Modal::Import(path)=>actions.push(UiAction::ImportProject(path)),
                Modal::Png{resolution,..}=>match self.choose(FileKind::Png,true){
                    Ok(Some(path))=>actions.push(UiAction::ExportPng{path,resolution:resolution.clamp(1,8192)}),Ok(None)=>{},Err(error)=>self.modal=Some(Modal::Error(error)),
                },
                Modal::Error(_)=>{},
            }
        }else{self.modal=Some(modal);}
        actions
    }
}
