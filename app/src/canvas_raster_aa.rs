//! Fixed 4x4 coverage samples, composited independently at edges. Interior
//! pixels stay uniform, avoiding a full 2048x2048 intermediate framebuffer.
use crate::document::{Primitive,ShapeKind};
use eframe::egui::{Color32,ColorImage};
const SIDE:usize=4;
const SAMPLES:usize=SIDE*SIDE;
const FULL:u16=u16::MAX;

pub(super) fn render(primitives:&[Primitive],size:usize)->ColorImage{
    assert!(size>0);
    let mut pixels=vec![[255.0f32;3];size*size];
    let mut edges=Vec::<[[f32;3];SAMPLES]>::new();
    let mut edge_at=vec![usize::MAX;size*size];
    let step=512./size as f32;
    for p in primitives{
        let color=p.color.map(f32::from);
        if p.shape==ShapeKind::Background{
            pixels.fill(color);edges.clear();edge_at.fill(usize::MAX);continue;
        }
        if p.width<=0.||p.height<=0.{continue;}
        let (sin,cos)=p.angle_deg.to_radians().sin_cos();
        let hw=p.width*0.5;let hh=p.height*0.5;
        let (ex,ey)=if p.shape==ShapeKind::Ellipse{((hw*cos).hypot(hh*sin),(hw*sin).hypot(hh*cos))}
            else{(hw*cos.abs()+hh*sin.abs(),hw*sin.abs()+hh*cos.abs())};
        let x0=((p.x-ex)/step).floor().clamp(0.,size as f32)as usize;
        let x1=((p.x+ex)/step).ceil().clamp(0.,size as f32)as usize;
        let y0=((p.y-ey)/step).floor().clamp(0.,size as f32)as usize;
        let y1=((p.y+ey)/step).ceil().clamp(0.,size as f32)as usize;
        let extent=(sin.abs()+cos.abs())*step*0.5;
        for y in y0..y1{for x in x0..x1{
            let dx=(x as f32+0.5)*step-p.x;let dy=(y as f32+0.5)*step-p.y;
            let lx=dx*cos+dy*sin;let ly=-dx*sin+dy*cos;
            let hi_x=lx.abs()+extent;let hi_y=ly.abs()+extent;
            let lo_x=(lx.abs()-extent).max(0.);let lo_y=(ly.abs()-extent).max(0.);
            let (inside,outside)=if p.shape==ShapeKind::Ellipse{
                ((hi_x/hw).powi(2)+(hi_y/hh).powi(2)<=1.,(lo_x/hw).powi(2)+(lo_y/hh).powi(2)>1.)
            }else{(hi_x<=hw&&hi_y<=hh,lo_x>hw||lo_y>hh)};
            if outside{continue;}
            let mask=if inside{FULL}else{
                let mut mask=0u16;
                for sy in 0..SIDE{for sx in 0..SIDE{
                    let ox=((sx as f32+0.5)/SIDE as f32-0.5)*step;
                    let oy=((sy as f32+0.5)/SIDE as f32-0.5)*step;
                    let ax=lx+ox*cos+oy*sin;let ay=ly-ox*sin+oy*cos;
                    let covered=if p.shape==ShapeKind::Ellipse{(ax/hw).powi(2)+(ay/hh).powi(2)<=1.}
                        else{ax.abs()<=hw&&ay.abs()<=hh};
                    if covered{mask|=1<<(sy*SIDE+sx);}
                }}mask
            };
            if mask==0{continue;}
            let index=y*size+x;
            if mask==FULL&&edge_at[index]==usize::MAX{
                for c in 0..3{pixels[index][c]=(pixels[index][c]+color[c])*0.5;}
            }else{
                if edge_at[index]==usize::MAX{edge_at[index]=edges.len();edges.push([pixels[index];SAMPLES]);}
                let samples=&mut edges[edge_at[index]];
                for (sample,rgb)in samples.iter_mut().enumerate(){if mask&(1<<sample)!=0{
                    for c in 0..3{rgb[c]=(rgb[c]+color[c])*0.5;}
                }}
            }
        }}
    }
    for (pixel,&edge)in pixels.iter_mut().zip(&edge_at){if edge!=usize::MAX{
        *pixel=[0.;3];for sample in &edges[edge]{for c in 0..3{pixel[c]+=sample[c]/SAMPLES as f32;}}
    }}
    ColorImage{size:[size,size],pixels:pixels.into_iter().map(|p|Color32::from_rgb(p[0].round()as u8,p[1].round()as u8,p[2].round()as u8)).collect()}
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn antialiases_edges_at_512_and_composites_overlapping_samples_before_resolving(){
        let bg=Primitive::background([255;3]);
        let p=Primitive::new(ShapeKind::Rectangle,0.5,0.5,0.5,1.,[0;3]);
        let image=render(&[bg.clone(),p.clone()],512);
        assert_eq!(image.size,[512,512]);assert_eq!(image[(0,0)],Color32::from_rgb(191,191,191));
        assert_eq!(image[(1,0)],Color32::WHITE);
        let layered=render(&[bg,p.clone(),p],512);
        assert_eq!(layered[(0,0)],Color32::from_rgb(159,159,159),"subpixel layers must composite before their average");
    }
    #[test]fn fully_covered_pixels_keep_native_colors_and_rotated_edges_get_partial_coverage(){
        let bg=Primitive::background([20,40,100]);
        let mut p=Primitive::new(ShapeKind::Ellipse,40.25,40.25,40.,20.,[220,80,0]);p.angle_deg=30.;
        let image=render(&[bg,p],512);let background=Color32::from_rgb(20,40,100);let solid=Color32::from_rgb(120,60,50);
        assert_eq!(image[(0,0)],background);assert_eq!(image[(40,40)],solid);
        assert!(image.pixels.iter().any(|&c|c!=background&&c!=solid),"edge pixels need fractional coverage");
    }
}
