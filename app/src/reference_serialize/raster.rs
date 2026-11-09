//! Incremental equivalent of the editor's 4x4 antialiased 512px display.
use crate::document::{Primitive,ShapeKind};
use image::{Rgb,RgbImage};
const SIDE:usize=4;
const SAMPLES:usize=16;

#[derive(Clone)]
pub(super) struct Raster{
    pixels:Vec<[f32;3]>,edge_at:Vec<usize>,edges:Vec<[[f32;3];SAMPLES]>,pub sse:f64,
}
impl Raster{
    pub fn new(bg:[u8;3],target:&RgbImage)->Self{
        let color=bg.map(f32::from);
        let mut result=Self{pixels:vec![color;512*512],edge_at:vec![usize::MAX;512*512],edges:Vec::new(),sse:0.};
        result.sse=target.pixels().map(|p|(0..3).map(|c|(p[c]as f64-bg[c]as f64).powi(2)).sum::<f64>()).sum();result
    }
    fn rgb(&self,index:usize)->[u8;3]{
        let mut value=self.pixels[index];let edge=self.edge_at[index];
        if edge!=usize::MAX{value=[0.;3];for sample in &self.edges[edge]{for c in 0..3{value[c]+=sample[c]/16.;}}}
        value.map(|v|v.round().clamp(0.,255.)as u8)
    }
    pub fn rmse(&self)->f64{(self.sse.max(0.)/(512.*512.*3.)).sqrt()}
    pub fn preview(&self)->RgbImage{RgbImage::from_fn(512,512,|x,y|Rgb(self.rgb((y*512+x)as usize)))}
    pub fn packed(&self)->Vec<u32>{(0..512*512).map(|i|{let p=self.rgb(i);(u32::from(p[0])<<16)|(u32::from(p[1])<<8)|u32::from(p[2])}).collect()}
    pub fn region_sse_exact(&self,target:&RgbImage,side:usize,index:usize)->f64{
        assert!(matches!(side,4|8)&&index<side*side,"valid reference grid cell required");
        let width=512/side;let x0=(index%side)*width;let y0=(index/side)*width;
        let mut sse=0f64;for y in y0..y0+width{for x in x0..x0+width{let p=self.rgb(y*512+x);let t=target.get_pixel(x as u32,y as u32);
            for c in 0..3{sse+=(p[c]as f64-t[c]as f64).powi(2);}
        }}sse
    }
    pub fn region_sse(&self,target:&RgbImage,side:usize,index:usize)->f32{
        self.region_sse_exact(target,side,index)as f32
    }
    pub fn region_rmse(&self,target:&RgbImage,side:usize,index:usize)->f64{
        (self.region_sse_exact(target,side,index)/((512/side)*(512/side)*3)as f64).sqrt()
    }
    pub fn apply(&mut self,p:&Primitive,target:&RgbImage){
        if p.shape==ShapeKind::Background{*self=Self::new(p.color,target);return;}
        if p.width<=0.||p.height<=0.{return;}
        let color=p.color.map(f32::from);let(sin,cos)=p.angle_deg.to_radians().sin_cos();let hw=p.width/2.;let hh=p.height/2.;
        let(ex,ey)=if p.shape==ShapeKind::Ellipse{((hw*cos).hypot(hh*sin),(hw*sin).hypot(hh*cos))}
            else{(hw*cos.abs()+hh*sin.abs(),hw*sin.abs()+hh*cos.abs())};
        let x0=(p.x-ex).floor().clamp(0.,512.)as usize;let x1=(p.x+ex).ceil().clamp(0.,512.)as usize;
        let y0=(p.y-ey).floor().clamp(0.,512.)as usize;let y1=(p.y+ey).ceil().clamp(0.,512.)as usize;
        let extent=(sin.abs()+cos.abs())*0.5;
        for y in y0..y1{for x in x0..x1{
            let dx=x as f32+0.5-p.x;let dy=y as f32+0.5-p.y;let lx=dx*cos+dy*sin;let ly=-dx*sin+dy*cos;
            let hx=lx.abs()+extent;let hy=ly.abs()+extent;let ax=(lx.abs()-extent).max(0.);let ay=(ly.abs()-extent).max(0.);
            let(inside,outside)=if p.shape==ShapeKind::Ellipse{((hx/hw).powi(2)+(hy/hh).powi(2)<=1.,(ax/hw).powi(2)+(ay/hh).powi(2)>1.)}
                else{(hx<=hw&&hy<=hh,ax>hw||ay>hh)};
            if outside{continue;}
            let mask=if inside{u16::MAX}else{let mut mask=0u16;for sy in 0..SIDE{for sx in 0..SIDE{
                let ox=(sx as f32+0.5)/4.-0.5;let oy=(sy as f32+0.5)/4.-0.5;
                let a=lx+ox*cos+oy*sin;let b=ly-ox*sin+oy*cos;
                let yes=if p.shape==ShapeKind::Ellipse{(a/hw).powi(2)+(b/hh).powi(2)<=1.}else{a.abs()<=hw&&b.abs()<=hh};
                if yes{mask|=1<<(sy*4+sx);}
            }}mask};
            if mask==0{continue;}let index=y*512+x;let before=self.rgb(index);
            if mask==u16::MAX&&self.edge_at[index]==usize::MAX{for c in 0..3{self.pixels[index][c]=(self.pixels[index][c]+color[c])*0.5;}}
            else{
                if self.edge_at[index]==usize::MAX{self.edge_at[index]=self.edges.len();self.edges.push([self.pixels[index];16]);}
                for(sample,rgb)in self.edges[self.edge_at[index]].iter_mut().enumerate(){if mask&(1<<sample)!=0{for c in 0..3{rgb[c]=(rgb[c]+color[c])*0.5;}}}
            }
            let after=self.rgb(index);let t=target.get_pixel(x as u32,y as u32);
            for c in 0..3{self.sse+=(after[c]as f64-t[c]as f64).powi(2)-(before[c]as f64-t[c]as f64).powi(2);}
        }}self.sse=self.sse.max(0.);
    }
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn incremental_quantized_raster_equals_editor_antialias_and_exact_rmse(){
        let target=RgbImage::from_fn(512,512,|x,y|Rgb([(x%256)as u8,(y%256)as u8,120]));
        let bg=Primitive::background([20,40,60]).normalized().unwrap();let mut raster=Raster::new(bg.color,&target);let mut all=vec![bg];
        for n in 0..18{let mut p=Primitive::new(if n%2==0{ShapeKind::Rectangle}else{ShapeKind::Ellipse},
            20.+n as f32*25.,60.+n as f32*19.,40.+n as f32,20.+n as f32,[20+n*10,180,80]);p.angle_deg=n as f32*7.2;
            let p=p.normalized().unwrap();raster.apply(&p,&target);all.push(p);
        }
        let image=crate::canvas::render_reference_preview(&all);let preview=raster.preview();
        assert_eq!(image.size,[512,512]);
        for(pixel,color)in preview.pixels().zip(&image.pixels){assert_eq!(pixel.0,[color.r(),color.g(),color.b()]);}
        let exact=preview.pixels().zip(target.pixels()).map(|(p,t)|(0..3).map(|c|(p[c]as f64-t[c]as f64).powi(2)).sum::<f64>()).sum::<f64>();
        assert!((raster.sse-exact).abs()<0.1);
    }
}
