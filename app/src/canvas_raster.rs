//! A 512-pixel display surface; edits and interaction overlays remain vectors.
use crate::document::Primitive;
use eframe::egui::{self,Color32,Pos2,Rect};
#[path="canvas_raster_aa.rs"]
mod antialias;
pub(crate) fn render_reference_preview(primitives:&[Primitive])->egui::ColorImage{antialias::render(primitives,SIZE)}

pub(super) const SIZE:usize=512;

#[derive(Default)]
pub(super) struct RasterCanvas{
    source:Vec<Primitive>,
    texture:Option<egui::TextureHandle>,
    #[cfg(test)]updates:usize,
}

fn same_visual(a:&Primitive,b:&Primitive)->bool{
    a.shape==b.shape&&a.x==b.x&&a.y==b.y&&a.width==b.width&&a.height==b.height
        &&a.angle_deg==b.angle_deg&&a.color==b.color
}

impl RasterCanvas{
    fn ensure_texture(&mut self,ctx:&egui::Context,primitives:&[Primitive])->egui::TextureId{
        let changed=self.texture.is_none()||self.source.len()!=primitives.len()
            ||!self.source.iter().zip(primitives).all(|(a,b)|same_visual(a,b));
        if changed{
            let image=antialias::render(primitives,SIZE);
            if let Some(texture)=self.texture.as_mut(){texture.set(image,egui::TextureOptions::NEAREST);}
            else{self.texture=Some(ctx.load_texture("main-canvas-512-raster",image,egui::TextureOptions::NEAREST));}
            self.source.clear();self.source.extend_from_slice(primitives);
            #[cfg(test)]{self.updates+=1;}
        }
        self.texture.as_ref().expect("raster texture was initialized").id()
    }
    pub(super) fn paint(&mut self,ctx:&egui::Context,painter:&egui::Painter,board:Rect,primitives:&[Primitive]){
        let texture=self.ensure_texture(ctx,primitives);
        painter.image(texture,board,Rect::from_min_max(Pos2::ZERO,Pos2::new(1.,1.)),Color32::WHITE);
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::document::{PrimitiveStageAssignment,StageTier,ShapeKind};
    fn source()->Vec<Primitive>{
        let mut rect=Primitive::new(ShapeKind::Rectangle,0.5,0.5,1.,1.,[220,80,0]);rect.id=2;
        vec![Primitive::background([20,40,100]),rect]
    }
    fn frame(ctx:&egui::Context,raster:&mut RasterCanvas,primitives:&[Primitive])->egui::FullOutput{
        ctx.run(Default::default(),|ctx|{egui::CentralPanel::default().show(ctx,|ui|{
            raster.paint(ctx,ui.painter(),Rect::from_min_size(Pos2::ZERO,egui::Vec2::splat(2048.)),primitives);
        });})
    }
    #[test]
    fn surface_is_exactly_512_pixels_with_nearest_filter_and_native_half_alpha(){
        let ctx=egui::Context::default();let mut raster=RasterCanvas::default();let primitives=source();
        let output=frame(&ctx,&mut raster,&primitives);
        let (_,delta)=output.textures_delta.set.iter().find(|(id,_)|Some(*id)==raster.texture.as_ref().map(|t|t.id())).unwrap();
        assert_eq!(delta.image.size(),[512,512]);
        assert_eq!(delta.options.magnification,egui::TextureFilter::Nearest);
        assert_eq!(delta.options.minification,egui::TextureFilter::Nearest);
        let egui::ImageData::Color(image)=&delta.image else{panic!("canvas must use a color raster")};
        assert_eq!(image[(0,0)],Color32::from_rgb(120,60,50));
        assert_eq!(image[(1,0)],Color32::from_rgb(20,40,100));
        assert_eq!(image[(0,1)],Color32::from_rgb(20,40,100));
        assert_eq!(primitives,source(),"display cannot rewrite vector geometry");
    }
    #[test]
    fn idle_views_and_metadata_reuse_texture_but_visual_edits_and_order_refresh_it(){
        let ctx=egui::Context::default();let mut raster=RasterCanvas::default();let mut primitives=source();
        frame(&ctx,&mut raster,&primitives);let id=raster.texture.as_ref().unwrap().id();
        for _ in 0..4{assert!(frame(&ctx,&mut raster,&primitives).textures_delta.set.iter().all(|(t,_)|*t!=id));}
        primitives[1].id=900;
        primitives[1].stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Fine,cell:0,queue_order:3});
        frame(&ctx,&mut raster,&primitives);assert_eq!(raster.updates,1);
        primitives[1].x=1.5;frame(&ctx,&mut raster,&primitives);assert_eq!(raster.updates,2);
        primitives[1].color=[0,0,0];frame(&ctx,&mut raster,&primitives);assert_eq!(raster.updates,3);
        primitives.swap(0,1);frame(&ctx,&mut raster,&primitives);assert_eq!(raster.updates,4);
        assert_eq!(raster.texture.as_ref().unwrap().id(),id,"updates reuse the same GPU texture");
    }
}
