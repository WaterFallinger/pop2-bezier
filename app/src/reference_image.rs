//! Source images stay at their original resolution. Only the confirmed square
//! view is resampled to the model's 512x512 canvas.
use crate::ui::reference_ui::{PixelCrop,ReferenceImageView};
use image::{GenericImageView,Rgb,RgbImage,RgbaImage};
use std::{path::Path,sync::Arc};

pub struct SourceImage{pub rgba:RgbaImage}

pub fn decode(path:&Path)->Result<SourceImage,String>{
    let reader=image::io::Reader::open(path).map_err(|e|format!("无法读取参考图像：{e}"))?
        .with_guessed_format().map_err(|e|format!("无法识别参考图像格式：{e}"))?;
    let decoded=reader.decode().map_err(|e|format!("参考图像解码失败：{e}"))?;
    if decoded.width()==0||decoded.height()==0{return Err("参考图像没有有效像素".into());}
    Ok(SourceImage{rgba:decoded.into_rgba8()})
}

struct WhiteCrop<'a>{source:&'a RgbaImage,crop:PixelCrop}
impl GenericImageView for WhiteCrop<'_>{
    type Pixel=Rgb<u8>;
    fn dimensions(&self)->(u32,u32){(self.crop.size,self.crop.size)}
    fn bounds(&self)->(u32,u32,u32,u32){(0,0,self.crop.size,self.crop.size)}
    fn get_pixel(&self,x:u32,y:u32)->Rgb<u8>{
        let p=self.source.get_pixel(self.crop.x+x,self.crop.y+y).0;let alpha=u32::from(p[3]);
        Rgb(std::array::from_fn(|c|((u32::from(p[c])*alpha+255*(255-alpha)+127)/255)as u8))
    }
}

pub fn crop_to_target(source:&SourceImage,crop:&PixelCrop)->Result<RgbImage,String>{
    let dimensions=[source.rgba.width()as usize,source.rgba.height()as usize];
    if !crop.valid_for(dimensions){return Err("裁剪区域为空或超出原图范围，请重新框选".into());}
    let view=WhiteCrop{source:&source.rgba,crop:*crop};
    Ok(image::imageops::resize(&view,512,512,image::imageops::FilterType::Lanczos3))
}

pub fn source_to_view(source:&SourceImage,revision:u64,name:String)->ReferenceImageView{
    let (width,height)=source.rgba.dimensions();
    let preview=if width.max(height)>2048{
        let factor=2048.0/width.max(height)as f64;
        image::imageops::resize(&source.rgba,(width as f64*factor).round().max(1.)as u32,
            (height as f64*factor).round().max(1.)as u32,image::imageops::FilterType::Triangle)
    }else{source.rgba.clone()};
    ReferenceImageView{revision,name,size:[width as usize,height as usize],image:Arc::new(
        eframe::egui::ColorImage::from_rgba_unmultiplied([preview.width()as usize,preview.height()as usize],preview.as_raw()))}
}

pub fn rgb_to_view(image:&RgbImage,revision:u64,name:String)->ReferenceImageView{
    ReferenceImageView{revision,name,size:[image.width()as usize,image.height()as usize],image:Arc::new(
        eframe::egui::ColorImage::from_rgb([image.width()as usize,image.height()as usize],image.as_raw()))}
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn arbitrary_source_crop_size_is_resized_to_512_without_canvas_pixel_restriction(){
        for size in [1,17,512,900]{
            let source=SourceImage{rgba:RgbaImage::from_pixel(1100,1000,image::Rgba([20,80,160,255]))};
            let target=crop_to_target(&source,&PixelCrop{x:100,y:50,size}).unwrap();
            assert_eq!(target.dimensions(),(512,512));assert!(target.pixels().all(|p|p.0==[20,80,160]));
        }
    }
    #[test]fn crop_uses_original_coordinates_and_flattens_transparency_before_resampling(){
        let mut source=SourceImage{rgba:RgbaImage::from_pixel(1600,900,image::Rgba([0,0,0,0]))};
        for y in 40..140{for x in 1200..1300{source.rgba.put_pixel(x,y,image::Rgba([200,100,0,128]));}}
        let target=crop_to_target(&source,&PixelCrop{x:1200,y:40,size:100}).unwrap();
        assert!(target.pixels().all(|p|p.0==[227,177,127]));
        assert!(crop_to_target(&source,&PixelCrop{x:1200,y:40,size:0}).is_err());
        assert!(crop_to_target(&source,&PixelCrop{x:1600,y:0,size:1}).is_err());
        assert!(crop_to_target(&source,&PixelCrop{x:u32::MAX,y:0,size:2}).is_err());
    }
    #[test]fn source_preview_has_bounded_texture_but_preserves_original_crop_coordinates(){
        let source=SourceImage{rgba:RgbaImage::from_pixel(4096,1024,image::Rgba([20,40,60,255]))};
        let view=source_to_view(&source,9,"large.png".into());
        assert_eq!(view.size,[4096,1024]);assert_eq!(view.image.size,[2048,512]);assert_eq!(source.rgba.dimensions(),(4096,1024));
    }
}
