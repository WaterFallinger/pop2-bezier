//! Presentation lifecycle only. No model or document commands are issued here.
use crate::{compatibility_dialog::CompatibilityDialog, compatibility_notice::CompatibilityNotice, diagnostics};
use eframe::egui;
use serde_json::json;
use std::path::PathBuf;

pub struct CompatibilityStartup {
    notice:Option<CompatibilityNotice>,
    dialog:CompatibilityDialog,
    presented:bool,
    rendered:bool,
    render_frames:usize,
    dismissed:bool,
    copy_count:usize,
    nodes_before:Option<usize>,
    report:Option<PathBuf>,
    capture_notice:bool,
}
impl Default for CompatibilityStartup {
    fn default()->Self {
        let args=std::env::args().collect::<Vec<_>>();
        Self{notice:None,dialog:CompatibilityDialog::default(),presented:false,rendered:false,render_frames:0,
            dismissed:false,copy_count:0,nodes_before:None,
            report:args.iter().position(|a|a=="--compatibility-notice-report").and_then(|i|args.get(i+1)).map(PathBuf::from),
            capture_notice:args.iter().any(|a|a=="--capture-compatibility-notice")}
    }
}
impl CompatibilityStartup {
    pub fn present(&mut self,notice:CompatibilityNotice){
        if self.presented{return}
        self.notice=Some(notice);self.presented=true;self.dialog.open();
    }
    pub fn was_presented(&self)->bool{self.presented}
    pub fn is_open(&self)->bool{self.dialog.is_open()}
    pub fn capture_requested(&self)->bool{self.capture_notice&&self.is_open()&&self.render_frames>=3}
    pub fn snapshot(&self)->serde_json::Value{
        json!({"presented":self.presented,"visible":self.is_open(),"rendered":self.rendered,
            "dismissed":self.dismissed,"copy_count":self.copy_count,"notice":self.notice})
    }
    pub fn show(&mut self,ctx:&egui::Context,loaded:bool,running:bool,nodes:usize){
        let Some(notice)=&self.notice else{return};
        self.nodes_before.get_or_insert(nodes);
        let actions=self.dialog.show(ctx,notice);
        let first=!self.rendered;self.rendered=true;
        self.render_frames+=1;
        if actions.copied{self.copy_count+=1;diagnostics::event("compatibility_notice_copied",json!({"complete_summary":true}));}
        if actions.open_logs{
            if let Err(error)=diagnostics::open_directory(){diagnostics::error("compatibility_notice_open_logs",&error);}
        }
        if actions.dismissed{
            self.dismissed=true;
            diagnostics::event("compatibility_notice_dismissed",json!({"model_loaded":loaded,"model_running":running,"nodes_before":self.nodes_before,"nodes_after":nodes,"continue_using":true}));
            ctx.request_repaint();
        }
        if first||self.render_frames==3||actions.copied||actions.dismissed||actions.open_logs{
            if first{diagnostics::event("compatibility_notice_shown",json!({"backend":notice.current_backend,"scheme":notice.current_scheme,"disabled":notice.disabled_features}));}
            if let Some(path)=&self.report{
                let rect=|r:egui::Rect|json!({"min":[r.min.x,r.min.y],"max":[r.max.x,r.max.y]});
                let value=json!({"version":env!("CARGO_PKG_VERSION"),"presented":self.presented,
                    "visible":self.is_open(),"rendered":self.rendered,"dismissed":self.dismissed,
                    "copy_count":self.copy_count,"model_loaded":loaded,"generation_running":running,
                    "nodes_before":self.nodes_before,"nodes_after":nodes,"notice":notice,
                    "pixels_per_point":ctx.pixels_per_point(),"copy_rect":rect(self.dialog.copy_rect),
                    "continue_rect":rect(self.dialog.continue_rect),"close_rect":rect(self.dialog.close_rect)});
                if let Some(parent)=path.parent(){let _=std::fs::create_dir_all(parent);}
                if let Ok(bytes)=serde_json::to_vec_pretty(&value){let _=std::fs::write(path,bytes);}
            }
        }
    }
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn a_single_startup_notice_does_not_reopen_after_another_load(){
        let make=|reason:&str|CompatibilityNotice{current_backend:"CPU".into(),current_scheme:"CPU兼容".into(),reason:reason.into(),disabled_features:vec![],ai_summary:reason.into(),log_directory:None};
        let mut state=CompatibilityStartup::default();state.present(make("first"));state.present(make("second"));
        assert!(state.is_open());assert_eq!(state.notice.as_ref().unwrap().reason,"first");
        assert_eq!(state.snapshot()["copy_count"],0);
    }
}
