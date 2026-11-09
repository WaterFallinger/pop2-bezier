//! Stage-separated read-only previews and region selection for local redraw.
//! Selection and locking are provided by the application, never inferred here.
use crate::{document::Primitive,region_state::Stage};
use eframe::egui::{self,Align,Color32,Pos2,Rect,RichText,Sense,Stroke,TextureHandle,Vec2};

const INK:Color32=Color32::from_rgb(34,43,62);
const MUTED:Color32=Color32::from_rgb(111,122,143);
const LINE:Color32=Color32::from_rgb(211,220,233);
const YELLOW:Color32=Color32::from_rgb(236,179,38);
const RED:Color32=Color32::from_rgb(222,75,80);

#[derive(Clone,Debug)]
pub struct RedrawUiModel{
    pub global:Vec<Primitive>,
    pub coarse:Vec<Primitive>,
    pub fine:Vec<Primitive>,
    pub coarse_counts:[usize;16],
    pub fine_counts:[usize;64],
    pub selected_global:bool,
    pub selected_coarse:Vec<usize>,
    /// Manual selections. Locked cells may also occur here; red takes priority.
    pub selected_fine:Vec<usize>,
    pub locked_coarse:Vec<usize>,
    pub locked_fine:Vec<usize>,
    pub can_generate:bool,
    pub can_continue_global:bool,
    pub reason:Option<String>,
}
impl Default for RedrawUiModel{
    fn default()->Self{Self{global:Vec::new(),coarse:Vec::new(),fine:Vec::new(),coarse_counts:[0;16],fine_counts:[0;64],
        selected_global:false,selected_coarse:Vec::new(),selected_fine:Vec::new(),locked_coarse:Vec::new(),locked_fine:Vec::new(),can_generate:false,can_continue_global:false,reason:None}}
}

#[derive(Default)]struct StagePreview{primitives:Option<Vec<Primitive>>,texture:Option<TextureHandle>}
impl StagePreview{
    fn texture(&mut self,ctx:&egui::Context,primitives:&[Primitive],name:&str)->egui::TextureId{
        if self.primitives.as_deref()!=Some(primitives){
            let image=crate::thumbnails::render_preview(primitives,192);
            if let Some(texture)=&mut self.texture{texture.set(image,egui::TextureOptions::LINEAR);}
            else{self.texture=Some(ctx.load_texture(name,image,egui::TextureOptions::LINEAR));}
            self.primitives=Some(primitives.to_vec());
        }
        self.texture.as_ref().expect("stage preview initialized").id()
    }
}

#[derive(Default)]struct SelectionStroke{
    last:Option<Pos2>,
    visited:std::collections::BTreeSet<(usize,usize)>,
    explained_lock:bool,
    preview_side:f32,
    layout_locked:bool,
}

pub(super) struct RedrawUi{
    pub(super) focused:bool,
    visible:[bool;3],
    boards:[Rect;3],
    full_boards:[Rect;3],
    stroke:Option<SelectionStroke>,
    previews:[StagePreview;3],
    #[cfg(test)]visibility_rects:[Rect;3],
    #[cfg(test)]clear_locked_rect:Rect,
    #[cfg(test)]allocated_boards:[Rect;3],
    #[cfg(test)]legend_rect:Rect,
}
impl Default for RedrawUi{
    fn default()->Self{Self{focused:false,visible:[true;3],boards:[Rect::NOTHING;3],full_boards:[Rect::NOTHING;3],stroke:None,previews:std::array::from_fn(|_|StagePreview::default()),
        #[cfg(test)]visibility_rects:[Rect::NOTHING;3],
        #[cfg(test)]clear_locked_rect:Rect::NOTHING,
        #[cfg(test)]allocated_boards:[Rect::NOTHING;3],#[cfg(test)]legend_rect:Rect::NOTHING}}
}
pub(super) struct PanelOutput{
    pub panel:Rect,
    pub toggles:Vec<(Stage,usize)>,
    pub locked_hint:Option<String>,
    pub clear_locked:bool,
}
impl RedrawUi{
    pub(super) fn clear_focus(&mut self){self.focused=false;self.stroke=None;}
    pub(super) fn reset(&mut self){let visible=self.visible;*self=Self::default();self.visible=visible;}
    pub(super) fn hidden_ids(&self,model:&RedrawUiModel)->std::collections::HashSet<u64>{
        [&model.global,&model.coarse,&model.fine].into_iter().enumerate()
            .filter(|(slot,_)|!self.visible[*slot]).flat_map(|(_,rows)|rows.iter().map(|p|p.id)).collect()
    }
    pub(super) fn visible(&self,stage:Stage)->bool{match stage{
        Stage::Global=>self.visible[0],Stage::Coarse=>self.visible[1],Stage::Fine=>self.visible[2],Stage::Complete=>true,
    }}
    #[cfg(test)]pub(super) fn visibility_rect(&self,stage:Stage)->Rect{match stage{
        Stage::Global=>self.visibility_rects[0],Stage::Coarse=>self.visibility_rects[1],Stage::Fine=>self.visibility_rects[2],Stage::Complete=>Rect::NOTHING,
    }}
    pub(super) fn contains_grid(&self,point:Pos2)->bool{self.boards.iter().any(|board|board.contains(point))}
    #[cfg(test)]pub(super) fn board_rect(&self,stage:Stage)->Rect{match stage{Stage::Global=>self.boards[0],Stage::Coarse=>self.boards[1],Stage::Fine=>self.boards[2],Stage::Complete=>Rect::NOTHING}}
    pub(super) fn show(&mut self,ui:&mut egui::Ui,model:&RedrawUiModel,enabled:bool)->PanelOutput{
        let area=ui.available_rect_before_wrap();
        let mut output=PanelOutput{panel:area,toggles:Vec::new(),locked_hint:None,clear_locked:false};
        self.boards=[Rect::NOTHING;3];self.full_boards=[Rect::NOTHING;3];
        let mut pointer_owners=[false;3];
        #[cfg(test)]{self.allocated_boards=[Rect::NOTHING;3];self.legend_rect=Rect::NOTHING;self.visibility_rects=[Rect::NOTHING;3];self.clear_locked_rect=Rect::NOTHING;}
        egui::Frame::none().fill(Color32::from_rgb(247,249,252)).inner_margin(12.).show(ui,|ui|{
            ui.set_max_width((area.width()-24.).max(60.));
            ui.label(RichText::new("局部修改").size(13.).strong().color(INK));
            ui.add_space(5.);
            let has_locked=!model.locked_coarse.is_empty()||!model.locked_fine.is_empty();
            // New dependency locks may appear after each emitted toggle.
            // Keep the boards stationary until this physical stroke ends.
            let layout_locked=self.stroke.as_ref().map_or(has_locked,|stroke|stroke.layout_locked);
            let bottom=if layout_locked{115.}else{85.};
            let side=self.stroke.as_ref().map_or_else(||((area.height()-if layout_locked{235.}else{205.})/3.).min(area.width()-28.).max(72.),|stroke|stroke.preview_side);
            egui::ScrollArea::vertical().id_source("redraw_stage_previews").auto_shrink([false,false])
                .max_height((area.height()-bottom).max(100.)).show(ui,|ui|{
                    ui.set_max_width((area.width()-24.).max(60.));
                    for(slot,stage,label,primitives)in [
                        (0,Stage::Global,"全图",model.global.as_slice()),
                        (1,Stage::Coarse,"粗网格",model.coarse.as_slice()),
                        (2,Stage::Fine,"细网格",model.fine.as_slice())]{
                        ui.vertical_centered(|ui|{
                            ui.horizontal(|ui|{
                                ui.label(RichText::new(label).size(12.).strong().color(INK));
                                ui.with_layout(egui::Layout::right_to_left(Align::Center),|ui|{
                                    let visible=ui.checkbox(&mut self.visible[slot],RichText::new("阶段可视化").size(11.));
                                    #[cfg(test)]{self.visibility_rects[slot]=visible.rect;}
                                    if visible.changed(){ui.ctx().request_repaint();}
                                    visible.on_hover_text("仅控制左侧画布中这一阶段的显示，不修改图元或重绘选择。");
                                });
                            });
                            let(size,_response)=ui.allocate_exact_size(Vec2::splat(side),Sense::hover());
                            self.full_boards[slot]=size;
                            #[cfg(test)]{self.allocated_boards[slot]=size;}
                            let texture=self.previews[slot].texture(ui.ctx(),primitives,match stage{
                                Stage::Global=>"redraw-global",Stage::Coarse=>"redraw-coarse",_=>"redraw-fine"});
                            let painter=ui.painter();painter.rect_filled(size,1.,Color32::WHITE);
                            painter.image(texture,size,Rect::from_min_max(Pos2::ZERO,Pos2::new(1.,1.)),Color32::WHITE);
                            if stage!=Stage::Global{
                                let grid=stage.side();let cell_size=size.width()/grid as f32;
                                let selected=if stage==Stage::Coarse{&model.selected_coarse}else{&model.selected_fine};
                                for index in 0..grid*grid{
                                    let cell=Rect::from_min_size(size.min+Vec2::new((index%grid)as f32,(index/grid)as f32)*cell_size,Vec2::splat(cell_size));
                                    let locked=if stage==Stage::Coarse{model.locked_coarse.contains(&index)}else{model.locked_fine.contains(&index)};
                                    if locked||selected.contains(&index){let color=if locked{RED}else{YELLOW};
                                        painter.rect_filled(cell.shrink(0.6),0.,Color32::from_rgba_unmultiplied(color.r(),color.g(),color.b(),65));
                                        painter.rect_stroke(cell.shrink(0.6),0.,Stroke::new(1.5,color));
                                        if locked&&cell_size>=17.{paint_lock(painter,cell.center(),cell_size.min(32.)*0.32);}
                                    }
                                }
                                for edge in 1..grid{let d=edge as f32*cell_size;
                                    painter.vline(size.left()+d,size.y_range(),Stroke::new(0.7,Color32::from_rgba_unmultiplied(65,80,105,110)));
                                    painter.hline(size.x_range(),size.top()+d,Stroke::new(0.7,Color32::from_rgba_unmultiplied(65,80,105,110)));}
                                self.boards[slot]=size.intersect(ui.clip_rect());
                                let response=ui.add_enabled_ui(enabled,|ui|ui.interact(size,egui::Id::new(("redraw_grid",slot)),Sense::click_and_drag())).inner;
                                pointer_owners[slot]=response.is_pointer_button_down_on()||response.clicked_by(egui::PointerButton::Primary);
                                if let Some(point)=response.hover_pos(){
                                    let column=((point.x-size.left())/cell_size).floor().max(0.)as usize;
                                    let row=((point.y-size.top())/cell_size).floor().max(0.)as usize;
                                    let index=row.min(grid-1)*grid+column.min(grid-1);
                                    let locked=if stage==Stage::Coarse{model.locked_coarse.contains(&index)}else{model.locked_fine.contains(&index)};
                                    let count=if stage==Stage::Coarse{model.coarse_counts[index]}else{model.fine_counts[index]};
                                    let hint=if locked{"开始生成前须手动清空。请先取消全图或对应父粗网格的选择，才能取消此格。"}
                                        else{"按住左键划过区域即可反选；黄色区域保留已有图元续画，按 Del 可手动清空。"};
                                    response.on_hover_text(format!("第{}行 · 第{}列 · {}个图元\n{}",row.min(grid-1)+1,column.min(grid-1)+1,count,hint));
                                }
                            }else{
                                self.boards[slot]=size.intersect(ui.clip_rect());
                                if model.selected_global{
                                    painter.rect_filled(size.shrink(0.6),0.,Color32::from_rgba_unmultiplied(YELLOW.r(),YELLOW.g(),YELLOW.b(),65));
                                    painter.rect_stroke(size.shrink(0.6),0.,Stroke::new(1.5,YELLOW));
                                }
                                let response=ui.add_enabled_ui(enabled,|ui|ui.interact(size,egui::Id::new(("redraw_grid",slot)),Sense::click_and_drag())).inner;
                                pointer_owners[slot]=response.is_pointer_button_down_on()||response.clicked_by(egui::PointerButton::Primary);
                                response.on_hover_text("按住左键划过可选择或取消全图阶段；已有全图图元会保留并续画，已有内容的粗、细网格会被红色锁定，须先手动清空。");
                            }
                            ui.painter().rect_stroke(size,1.,Stroke::new(1.,LINE));
                        });ui.add_space(8.);
                    }
                });
            ui.add_space(7.);let legend=ui.with_layout(egui::Layout::left_to_right(Align::Center),|ui|{
                legend(ui,YELLOW,"手动选择");ui.add_space(7.);legend(ui,RED,"随上级锁定");
            });
            #[cfg(test)]{self.legend_rect=legend.response.rect;}
            #[cfg(not(test))]let _=legend;
            if has_locked{
                ui.add_space(5.);let clear=ui.add_enabled(enabled,egui::Button::new("清空红色区域").min_size(Vec2::new(ui.available_width(),24.)));
                #[cfg(test)]{self.clear_locked_rect=clear.rect;}
                output.clear_locked=clear.clicked();
                clear.on_hover_text("仅删除红色锁定区域中的粗、细网格图元，保留黄色区域已有图元和所有选择。");
            }
            ui.add_space(5.);ui.label(RichText::new("Del 删除所选区域的同阶段图元").size(10.).color(MUTED));
        });
        self.select_by_stroke(ui,model,enabled,&pointer_owners,&mut output);
        output
    }

    fn select_by_stroke(&mut self,ui:&egui::Ui,model:&RedrawUiModel,enabled:bool,owners:&[bool;3],output:&mut PanelOutput){
        if !enabled{self.stroke=None;return;}
        let(events,down)=ui.input(|i|(i.events.clone(),i.pointer.primary_down()));
        for event in events{match event{
            egui::Event::PointerButton{pos,button:egui::PointerButton::Primary,pressed:true,..}=>{
                self.stroke=None;
                if self.boards.iter().enumerate().any(|(slot,board)|owners[slot]&&board.contains(pos)){
                    self.focused=true;self.stroke=Some(SelectionStroke{preview_side:self.full_boards[0].width(),
                        layout_locked:!model.locked_coarse.is_empty()||!model.locked_fine.is_empty(),..Default::default()});
                    self.sweep_to(pos,model,output);
                }
            },
            egui::Event::PointerMoved(pos)=>{self.sweep_to(pos,model,output);},
            egui::Event::PointerButton{pos,button:egui::PointerButton::Primary,pressed:false,..}=>{
                self.sweep_to(pos,model,output);self.stroke=None;
            },
            egui::Event::PointerGone|egui::Event::WindowFocused(false)=>{self.stroke=None;},
            _=>{},
        }}
        if !down{self.stroke=None;}
    }
    fn sweep_to(&mut self,pos:Pos2,model:&RedrawUiModel,output:&mut PanelOutput){
        let Some(stroke)=&mut self.stroke else{return;};
        let start=stroke.last.replace(pos).unwrap_or(pos);
        for(_,stage,index)in swept_regions(start,pos,&self.full_boards,&self.boards){
            // A region is inverted once per physical stroke, independently of
            // frame rate, stationary frames, or small pointer retracing.
            if !stroke.visited.insert((stage.side(),index)){continue;}
            let locked=match stage{Stage::Coarse=>model.locked_coarse.contains(&index),
                Stage::Fine=>model.locked_fine.contains(&index),_=>false};
            if locked{
                if !stroke.explained_lock{output.locked_hint=Some(if stage==Stage::Coarse{
                    "这个粗网格随全图阶段锁定，无法反选；请先取消全图选择。".into()
                }else{"红色细网格由上级阶段锁定，无法反选；请先取消对应上级的选择。".into()});stroke.explained_lock=true;}
            }else{output.toggles.push((stage,index));}
        }
    }
}

/// Intersect each pointer segment with the visible grid. Split at grid edges
/// so fast motion does not skip cells; a diagonal corner alone is not a visit.
fn swept_regions(from:Pos2,to:Pos2,full:&[Rect;3],clips:&[Rect;3])->Vec<(f32,Stage,usize)>{
    let mut hits=Vec::new();let delta=to-from;
    for(slot,stage)in [Stage::Global,Stage::Coarse,Stage::Fine].into_iter().enumerate(){
        let board=full[slot];let clip=clips[slot];if !board.is_positive()||!clip.is_positive(){continue;}
        let(mut enter,mut exit)=(0.0f32,1.0f32);let mut intersects=true;
        for(origin,d,low,high)in[(from.x,delta.x,clip.left(),clip.right()),(from.y,delta.y,clip.top(),clip.bottom())]{
            if d.abs()<f32::EPSILON{if origin<low||origin>high{intersects=false;break;}}
            else{let a=(low-origin)/d;let b=(high-origin)/d;enter=enter.max(a.min(b));exit=exit.min(a.max(b));}
        }
        if !intersects||enter>exit{continue;}
        let side=stage.side();let step=board.width()/side as f32;let mut times=vec![enter,exit];
        for edge in 1..side{
            if delta.x.abs()>=f32::EPSILON{let t=(board.left()+edge as f32*step-from.x)/delta.x;if t>enter&&t<exit{times.push(t);}}
            if delta.y.abs()>=f32::EPSILON{let t=(board.top()+edge as f32*step-from.y)/delta.y;if t>enter&&t<exit{times.push(t);}}
        }
        times.sort_unstable_by(f32::total_cmp);times.dedup();
        let index_at=|t:f32|{let p=from+delta*t;
            let x=((p.x-board.left())/step).floor().max(0.)as usize;
            let y=((p.y-board.top())/step).floor().max(0.)as usize;y.min(side-1)*side+x.min(side-1)};
        if clip.contains(from){hits.push((enter,stage,index_at(enter)));}
        for pair in times.windows(2){if pair[1]>pair[0]{let middle=(pair[0]+pair[1])*0.5;hits.push((middle,stage,index_at(middle)));}}
        if clip.contains(to){hits.push((exit,stage,index_at(exit)));}
    }
    hits.sort_by(|a,b|a.0.total_cmp(&b.0));hits
}

fn legend(ui:&mut egui::Ui,color:Color32,text:&str){
    let(rect,_)=ui.allocate_exact_size(Vec2::splat(8.),Sense::hover());ui.painter().rect_filled(rect,1.,color);
    ui.label(RichText::new(text).size(10.).color(MUTED));
}
fn paint_lock(painter:&egui::Painter,center:Pos2,size:f32){
    let body=Rect::from_center_size(center+Vec2::new(0.,size*0.2),Vec2::new(size,size*0.7));
    painter.rect_filled(body,1.3,Color32::from_rgb(188,48,55));
    let top=center-Vec2::new(0.,size*0.25);painter.circle_stroke(top,size*0.32,Stroke::new(1.2,Color32::from_rgb(188,48,55)));
}

#[cfg(test)]
mod tests{
    use super::*;
    fn frame(ctx:&egui::Context,state:&mut RedrawUi,model:&RedrawUiModel,events:Vec<egui::Event>)->PanelOutput{
        frame_enabled(ctx,state,model,events,true)
    }
    fn frame_enabled(ctx:&egui::Context,state:&mut RedrawUi,model:&RedrawUiModel,events:Vec<egui::Event>,enabled:bool)->PanelOutput{
        let mut output=None;let _=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(300.,1050.))),events,..Default::default()},|ctx|{
            egui::CentralPanel::default().show(ctx,|ui|output=Some(state.show(ui,model,enabled)));
        });output.unwrap()
    }
    fn click(ctx:&egui::Context,state:&mut RedrawUi,model:&RedrawUiModel,point:Pos2)->PanelOutput{
        let event=|pressed|egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()};
        let mut output=frame(ctx,state,model,vec![egui::Event::PointerMoved(point),event(true)]);
        let release=frame(ctx,state,model,vec![event(false)]);output.toggles.extend(release.toggles);
        if release.locked_hint.is_some(){output.locked_hint=release.locked_hint;}
        output.clear_locked|=release.clear_locked;output
    }
    fn pointer(pos:Pos2,pressed:bool)->egui::Event{
        egui::Event::PointerButton{pos,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()}
    }
    fn cell_point(board:Rect,side:usize,index:usize)->Pos2{
        board.min+Vec2::new((index%side)as f32+0.5,(index/side)as f32+0.5)*(board.width()/side as f32)
    }
    #[test]fn held_sweep_inverts_selected_and_unselected_cells_once_and_fills_fast_motion(){
        let ctx=egui::Context::default();let mut state=RedrawUi::default();
        let model=RedrawUiModel{selected_coarse:vec![1,3],..Default::default()};
        for _ in 0..2{frame(&ctx,&mut state,&model,vec![]);}
        let board=state.board_rect(Stage::Coarse);let a=cell_point(board,4,0);let b=cell_point(board,4,3);
        let start=frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(a),pointer(a,true)]);
        assert_eq!(start.toggles,[(Stage::Coarse,0)]);
        let moved=frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(b)]);
        assert_eq!(moved.toggles,[(Stage::Coarse,1),(Stage::Coarse,2),(Stage::Coarse,3)]);
        let mut selected=std::collections::BTreeSet::from([1,3]);
        for(_,index)in start.toggles.into_iter().chain(moved.toggles){if !selected.remove(&index){selected.insert(index);}}
        assert_eq!(selected,std::collections::BTreeSet::from([0,2]));
        assert!(frame(&ctx,&mut state,&model,vec![]).toggles.is_empty());
        assert!(frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(a)]).toggles.is_empty());
        assert!(frame(&ctx,&mut state,&model,vec![pointer(a,false)]).toggles.is_empty());
        assert_eq!(click(&ctx,&mut state,&model,a).toggles,[(Stage::Coarse,0)]);
        assert_eq!(state.board_rect(Stage::Coarse),board,"painting a selection must not scroll the preview");
    }
    #[test]fn one_frame_motion_preserves_bends_and_locked_cells_never_toggle(){
        let ctx=egui::Context::default();let mut state=RedrawUi::default();
        let model=RedrawUiModel{locked_fine:vec![1,3],..Default::default()};
        for _ in 0..2{frame(&ctx,&mut state,&model,vec![]);}
        let board=state.board_rect(Stage::Fine);let a=cell_point(board,8,0);let b=cell_point(board,8,4);let c=cell_point(board,8,20);
        assert_eq!(frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(a),pointer(a,true)]).toggles,[(Stage::Fine,0)]);
        let output=frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(b),egui::Event::PointerMoved(c)]);
        assert_eq!(output.toggles,[(Stage::Fine,2),(Stage::Fine,4),(Stage::Fine,12),(Stage::Fine,20)]);
        assert!(output.locked_hint.is_some());
        let output=frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(a)]);
        assert!(output.locked_hint.is_none(),"one explanation per stroke");
        frame(&ctx,&mut state,&model,vec![pointer(a,false)]);
    }
    #[test]fn selection_drag_must_start_on_a_preview_and_is_cancelled_when_disabled_or_pointer_is_lost(){
        let ctx=egui::Context::default();let mut state=RedrawUi::default();let model=RedrawUiModel::default();
        for _ in 0..2{frame(&ctx,&mut state,&model,vec![]);}
        let board=state.board_rect(Stage::Coarse);let a=cell_point(board,4,0);let b=cell_point(board,4,1);
        let outside=Pos2::new(1.,1.);
        assert!(frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(outside),pointer(outside,true)]).toggles.is_empty());
        assert!(frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(a)]).toggles.is_empty());
        frame(&ctx,&mut state,&model,vec![pointer(a,false)]);
        assert_eq!(frame(&ctx,&mut state,&model,vec![pointer(a,true)]).toggles,[(Stage::Coarse,0)]);
        assert!(frame_enabled(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(b)],false).toggles.is_empty());
        assert!(frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(a)]).toggles.is_empty());
        frame(&ctx,&mut state,&model,vec![pointer(a,false)]);
        for _ in 0..2{frame(&ctx,&mut state,&model,vec![]);}
        assert_eq!(frame(&ctx,&mut state,&model,vec![pointer(a,true)]).toggles,[(Stage::Coarse,0)]);
        frame(&ctx,&mut state,&model,vec![egui::Event::PointerGone]);
        assert!(frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(b)]).toggles.is_empty());
    }
    #[test]fn swept_grid_clips_to_visible_cells_and_does_not_select_diagonal_neighbours(){
        let full=[Rect::NOTHING,Rect::from_min_size(Pos2::ZERO,Vec2::splat(400.)),Rect::NOTHING];
        let mut clips=full;clips[1]=Rect::from_min_max(Pos2::new(0.,100.),Pos2::new(400.,300.));
        let cells=|a,b|swept_regions(a,b,&full,&clips).into_iter().map(|(_,_,index)|index).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(cells(Pos2::new(-50.,150.),Pos2::new(450.,150.)),std::collections::BTreeSet::from([4,5,6,7]));
        assert_eq!(cells(Pos2::new(50.,50.),Pos2::new(350.,350.)),std::collections::BTreeSet::from([5,10]));
        assert!(cells(Pos2::new(20.,50.),Pos2::new(380.,50.)).is_empty());
    }
    #[test]fn new_dependency_locks_do_not_shift_the_grid_under_a_held_stroke(){
        let ctx=egui::Context::default();let mut state=RedrawUi::default();let mut model=RedrawUiModel::default();
        for _ in 0..2{frame(&ctx,&mut state,&model,vec![]);}
        let board=state.board_rect(Stage::Coarse);let a=cell_point(board,4,0);let b=cell_point(board,4,3);
        assert_eq!(frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(a),pointer(a,true)]).toggles,[(Stage::Coarse,0)]);
        model.selected_coarse=vec![0];model.locked_fine=vec![0];
        assert!(frame(&ctx,&mut state,&model,vec![]).toggles.is_empty());
        assert_eq!(state.board_rect(Stage::Coarse),board);
        let output=frame(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(b)]);
        assert_eq!(output.toggles,[(Stage::Coarse,1),(Stage::Coarse,2),(Stage::Coarse,3)]);
        assert_eq!(state.board_rect(Stage::Coarse),board);
        frame(&ctx,&mut state,&model,vec![pointer(b,false)]);
    }
    #[test]fn all_three_stages_emit_toggles_while_locked_cells_only_explain(){
        let ctx=egui::Context::default();let mut state=RedrawUi::default();let model=RedrawUiModel{locked_fine:vec![0],selected_fine:vec![0],..Default::default()};
        assert!(frame(&ctx,&mut state,&model,vec![]).toggles.is_empty());
        let global=state.boards[0].center();assert_eq!(click(&ctx,&mut state,&model,global).toggles,vec![(Stage::Global,0)]);
        let coarse=state.boards[1];let point=coarse.min+Vec2::splat(coarse.width()/8.);
        assert_eq!(click(&ctx,&mut state,&model,point).toggles,vec![(Stage::Coarse,0)]);assert!(state.focused);
        let fine=state.boards[2];let point=fine.min+Vec2::splat(fine.width()/16.);
        let result=click(&ctx,&mut state,&model,point);assert!(result.toggles.is_empty());assert!(result.locked_hint.is_some());
    }
    #[test]fn global_selection_is_editable_only_when_idle_and_focuses_the_delete_target(){
        let ctx=egui::Context::default();let mut state=RedrawUi::default();
        let model=RedrawUiModel{selected_global:true,..Default::default()};
        for _ in 0..2{frame_enabled(&ctx,&mut state,&model,vec![],false);}
        let point=state.board_rect(Stage::Global).center();assert!(state.contains_grid(point));
        let event=|pressed|egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()};
        for pressed in [true,false]{let output=frame_enabled(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(point),event(pressed)],false);
            assert!(output.toggles.is_empty());}
        assert!(!state.focused);
        for _ in 0..2{frame(&ctx,&mut state,&model,vec![]);}
        let point=state.board_rect(Stage::Global).center();
        assert_eq!(click(&ctx,&mut state,&model,point).toggles,vec![(Stage::Global,0)]);assert!(state.focused);
        assert!(model.selected_global,"UI only emits a request; the application owns selection");
    }
    #[test]fn locked_coarse_cell_cannot_be_toggled_and_alone_enables_explicit_red_cleanup(){
        let ctx=egui::Context::default();let mut state=RedrawUi::default();
        let model=RedrawUiModel{selected_global:true,selected_coarse:vec![0],locked_coarse:vec![0],..Default::default()};
        for _ in 0..2{frame(&ctx,&mut state,&model,vec![]);}
        let coarse=state.board_rect(Stage::Coarse);let point=coarse.min+Vec2::splat(coarse.width()/8.);
        let output=click(&ctx,&mut state,&model,point);assert!(output.toggles.is_empty());
        assert!(output.locked_hint.is_some_and(|hint|hint.contains("全图")));
        assert!(state.clear_locked_rect.is_positive());let point=state.clear_locked_rect.center();
        let output=click(&ctx,&mut state,&model,point);assert!(output.clear_locked);assert!(output.toggles.is_empty());
        assert_eq!(model.selected_coarse,[0]);assert_eq!(model.locked_coarse,[0]);
    }
    #[test]fn stage_preview_texture_is_reused_for_selection_changes_and_identical_geometry(){
        let ctx=egui::Context::default();let mut preview=StagePreview::default();
        let a=preview.texture(&ctx,&[],"empty stage");let b=preview.texture(&ctx,&[],"empty stage");assert_eq!(a,b);
        assert!(preview.primitives.as_ref().unwrap().is_empty());
    }
    #[test]fn independent_visibility_switches_remain_available_with_busy_grids_and_reset_preserves_them(){
        use crate::document::ShapeKind;
        let ctx=egui::Context::default();let mut state=RedrawUi::default();
        let mut global=Primitive::background([230;3]);global.id=1;
        let mut coarse=Primitive::new(ShapeKind::Rectangle,40.,40.,12.,10.,[80,120,160]);coarse.id=2;
        let mut fine=coarse.clone();fine.id=3;
        let model=RedrawUiModel{global:vec![global],coarse:vec![coarse],fine:vec![fine],
            selected_coarse:vec![0],selected_fine:vec![1],locked_fine:vec![2],..Default::default()};
        for _ in 0..3{frame_enabled(&ctx,&mut state,&model,vec![],false);}
        assert!(state.hidden_ids(&model).is_empty());
        for(stage,id)in [(Stage::Global,1),(Stage::Coarse,2),(Stage::Fine,3)]{
            assert!(state.visible(stage));let point=state.visibility_rect(stage).center();
            let event=|pressed|egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()};
            for pressed in [true,false]{let output=frame_enabled(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(point),event(pressed)],false);
                assert!(output.toggles.is_empty());assert!(output.locked_hint.is_none());}
            assert!(!state.visible(stage));assert!(state.hidden_ids(&model).contains(&id));
        }
        assert_eq!(state.hidden_ids(&model).len(),3);
        state.reset();assert!(!state.visible(Stage::Global)&&!state.visible(Stage::Coarse)&&!state.visible(Stage::Fine));
        assert_eq!(model.selected_coarse,[0]);assert_eq!(model.selected_fine,[1]);assert_eq!(model.locked_fine,[2]);
        assert_eq!(model.coarse[0].id,2);assert_eq!(model.fine[0].id,3);
    }
    #[test]fn explicit_red_cleanup_button_only_emits_one_manual_action_when_grid_edits_are_enabled(){
        let ctx=egui::Context::default();let mut state=RedrawUi::default();
        let model=RedrawUiModel{selected_coarse:vec![0],locked_fine:vec![0],..Default::default()};
        for _ in 0..3{frame_enabled(&ctx,&mut state,&model,vec![],false);}
        let point=state.clear_locked_rect.center();assert!(state.clear_locked_rect.is_positive());
        let event=|pressed|egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()};
        for pressed in [true,false]{let output=frame_enabled(&ctx,&mut state,&model,vec![egui::Event::PointerMoved(point),event(pressed)],false);assert!(!output.clear_locked);}
        // Egui routes pointer presses using the preceding frame's widget map.
        // Publish the enabled button before sending a fresh pointer press.
        for _ in 0..2{frame(&ctx,&mut state,&model,vec![]);}
        let point=state.clear_locked_rect.center();
        let output=click(&ctx,&mut state,&model,point);assert!(output.clear_locked);assert!(output.toggles.is_empty());
        assert!(!frame(&ctx,&mut state,&model,vec![]).clear_locked,"red cleanup is never automatic");
        assert_eq!(model.selected_coarse,[0]);assert_eq!(model.locked_fine,[0]);
    }
    #[test]fn all_three_full_stage_boards_fit_above_the_legend_at_supported_editor_sizes(){
        use crate::{document::Document,region_state::DrawingRegion,ui::{EditorUi,NodeView,UiModel}};
        for size in [Vec2::new(1440.,920.),Vec2::new(1100.,720.)]{for locked in [false,true]{
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let doc=Document::new();let region=DrawingRegion::default();
            editor.set_save_name(Some("局部重绘布局验收".into()));editor.set_redraw_model(Some(RedrawUiModel{
                selected_coarse:if locked{vec![0]}else{vec![]},locked_fine:if locked{vec![0]}else{vec![]},..Default::default()}));
            let nodes=vec![NodeView{id:0,parent:None,title:"初始".into(),source:"初始".into(),primitive_count:0}];
            let model=UiModel{document:&doc,drawing_region:&region,primitives:&doc.primitives,stage_by_id:None,prompt:&doc.prompt,settings:&doc.settings,
                current_node:0,nodes:&nodes,running:false,loading:false,compiling:false,stopping:false,model_loaded:false,
                model_status:"模型未加载",model_path:"",tokenizer_path:concat!(env!("CARGO_MANIFEST_DIR"),"/assets/caption_spm_1500.model"),error:None,status:"就绪"};
            for _ in 0..2{let _=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,size)),..Default::default()},|ctx|{editor.render(ctx,&model);});}
            let panel=&editor.redraw_panel;
            for slot in 0..3{
                let allocated=panel.allocated_boards[slot];let visible=panel.boards[slot];
                assert!(allocated.is_positive(),"missing stage board {slot} at {size:?}");
                assert!((allocated.width()-allocated.height()).abs()<0.1);
                assert!(visible.contains_rect(allocated),"stage {slot} is clipped at {size:?}: full={allocated:?}, visible={visible:?}");
                assert!(allocated.bottom()<panel.legend_rect.top(),"stage {slot} overlaps legend at {size:?}");
            }
            assert!(panel.legend_rect.bottom()<size.y-15.);
        }}
    }
}
