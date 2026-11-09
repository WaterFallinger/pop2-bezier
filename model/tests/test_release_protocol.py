"""CPU-only release protocol checks; never loads the full 1B model."""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from token_layout import TOKEN_LAYOUT as L
from block_state import canonical_bos_tokens,build_branched_block_batch
from model_gpt import GeometrizeGPT

class ReleaseProtocolTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(17)
        return GeometrizeGPT(d_model=64,n_layer=2,n_head=4,ffn_hidden=128,max_context_len=64,max_block_positions=64,max_text_tokens=32)
    def records(self):
        bos=canonical_bos_tokens(1,'cpu')
        shape=torch.tensor([[256,1280,2048,2518,4566,6415,6702,6830,6958]],dtype=torch.long)
        return torch.cat([bos,shape,shape+torch.tensor([[4,4,0,0,0,0,0,0,0]])]).unsqueeze(0)
    def test_shared_history_reference_backward_and_causal_prefix(self):
        model=self.model();full=self.records();batch=build_branched_block_batch(full,1,rand_k=2,sampling_seed=19)
        logits=model.forward_multi_lane_reference(batch.full_tokens,batch.query_tokens,batch.query_block_ids,batch.query_fields,batch.first_query_block)
        loss=logits.float().square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            a=model(full);changed=full.clone();changed[:,-1,0]+=20;b=model(changed)
        torch.testing.assert_close(a[:,:-1],b[:,:-1],rtol=0,atol=0)
    def test_real_decoder_small_model_background_shape_and_bounds(self):
        import reference_visualize as v
        model=self.model().eval();old=(v.TARGET_SHAPE_STEPS,v.START_STEPS,v.ENABLE_OPTIMIZED_INFERENCE,v.RENYI_ALPHA)
        try:
            v.TARGET_SHAPE_STEPS=5;v.START_STEPS=1;v.ENABLE_OPTIMIZED_INFERENCE=False;v.RENYI_ALPHA=4
            text=torch.tensor([[2,3]]);mask=torch.ones_like(text,dtype=torch.bool)
            with torch.inference_mode():out=v.generate_candidate_batch(model,canonical_bos_tokens(1,'cpu')[:,None,:],'cpu',np.random.default_rng(21),intervention_policy='normal',text_ids=text,text_mask=mask)
            records=out[0][0].reshape(-1,9)
            self.assertEqual(records.shape,(5,9));self.assertEqual(int(records[1,5]),L.shape_offset)
            self.assertTrue(all(int(row[5])-L.shape_offset in (1,2) for row in records[2:]))
            self.assertEqual(v.rasterize_render_data(v.decode_tokens_to_render_data(records.reshape(-1))).shape,(512,512,3))
        finally:v.TARGET_SHAPE_STEPS,v.START_STEPS,v.ENABLE_OPTIMIZED_INFERENCE,v.RENYI_ALPHA=old
    def test_cfg_baseline_is_exact_and_residual_matches_formula(self):
        from cfg import GuidedModel
        z=torch.tensor([[1.,2.,3.],[0.,4.,1.]])
        guided=GuidedModel(None,1.);torch.testing.assert_close(guided.combine(z,1),z[:1],rtol=0,atol=0)
        guided.scale=2.;torch.testing.assert_close(guided.combine(z,1),z[1:]+2*(z[:1]-z[1:]))
    def test_default_checkpoint_retention_never_waits_for_cloud_ack_or_deletes(self):
        import train_gpt_pretrain as trainer
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'geometrize_gpt_pretrain_epoch_1.ckpt';old.mkdir();(old/'model.pt').write_bytes(b'complete-old')
            with patch.dict(os.environ,{},clear=True):trainer.reserve_three_a800_checkpoint_slot(root,3)
            self.assertEqual((old/'model.pt').read_bytes(),b'complete-old')
            with patch.dict(os.environ,{'POP_REQUIRE_ARCHIVE_ACK':'1'},clear=True):
                with self.assertRaisesRegex(ValueError,'V33_TRAINING_CONTROL_DIR'):trainer.reserve_three_a800_checkpoint_slot(root,3)
    def test_opt_in_ack_gate_preserves_unacknowledged_checkpoint(self):
        import train_gpt_pretrain as trainer
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'geometrize_gpt_pretrain_epoch_1.ckpt';old.mkdir();(old/'model.pt').write_bytes(b'old')
            rows=[((2,True,-1),root/'new.ckpt',{}),((1,True,-1),old,{'batch_idx':-1,'file_sizes':{'model.pt':3}})]
            with patch.dict(os.environ,{'POP_REQUIRE_ARCHIVE_ACK':'1','V33_TRAINING_CONTROL_DIR':str(root/'control')},clear=True),patch.object(trainer,'completed_sharded_checkpoints',return_value=rows),patch.object(trainer.time,'sleep',side_effect=RuntimeError('test intercepted safe wait')):
                with self.assertRaisesRegex(RuntimeError,'safe wait'):trainer.reserve_three_a800_checkpoint_slot(root,3)
            self.assertTrue((old/'model.pt').is_file())
    def test_caption_cardinality_is_not_fixed_to_private_100k(self):
        from caption_data import selected_rows,build_caption_cache,CaptionStore
        from config import TrainConfig
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'prompts/000').mkdir(parents=True)
            (root/'selected.tsv').write_text('priority\timage_id\tfilename\n0\t1\tsynthetic.png\n',encoding='utf-8')
            record={'filename':'synthetic.png','prompts':[{'variant':1,'text':'金发。'},{'variant':2,'text':'蓝眼。'}]}
            (root/'prompts/000/000001.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
            config=TrainConfig();config.CAPTION_SELECTION_PATH=str(root/'selected.tsv');config.CAPTION_PROMPTS_DIR=str(root/'prompts');config.CAPTION_CACHE_DIR=str(root/'cache');config.CAPTION_VARIANTS=2;config.TEXT_DROPOUT_PROBABILITY=0
            self.assertEqual(len(selected_rows(config.CAPTION_SELECTION_PATH)[0]),1)
            build_caption_cache(config);store=CaptionStore(config);ids,mask=store.batch(['synthetic'],1,42)
            self.assertEqual(ids.shape[0],1);self.assertEqual(int(ids[0,0]),2);self.assertEqual(int(ids[0,mask[0].sum()-1]),3)
            del store
            import gc;gc.collect()
            record['prompts'][0]['text']='金发'*300
            (root/'prompts/000/000001.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'224'):build_caption_cache(config)
    def test_consolidated_resume_alignment_is_general_and_shards_remain_strict(self):
        import train_gpt_pretrain as trainer
        from config import TrainConfig
        config=TrainConfig();config.MICRO_BATCH_SIZE=2;config.GRADIENT_ACCUMULATION_STEPS=75
        checkpoint={'optimizer_saved_world_size':3,'micro_batch_size_per_device':2,'gradient_accumulation_steps_per_device':100,'optimizer_state_format':trainer.CONSOLIDATED_OPTIMIZER_FORMAT}
        batch,loss=trainer.align_resume_batch(checkpoint,6099,10.,4,config)
        self.assertEqual(batch,4574);self.assertEqual(loss,7.5)
        checkpoint['optimizer_state_format']=trainer.GPU_MASTER_SHARDED_FORMAT
        with self.assertRaisesRegex(RuntimeError,'resharding'):trainer.align_resume_batch(checkpoint,-1,10.,4,config)

    def test_public_tiny_resume_selects_latest_full_checkpoint_path(self):
        import train
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{},clear=True):
            root=Path(temp);checkpoints=root/'checkpoints';checkpoints.mkdir()
            for epoch in (1,2):(checkpoints/f'geometrize_gpt_pretrain_epoch_{epoch}.pt').write_bytes(b'fixture')
            args=train.parser().parse_args(['--data-dir',str(root),'--caption-root',str(root),'--checkpoint-dir',str(checkpoints),'--resume-dir',str(checkpoints),'--tiny'])
            config=train.configure(args)
            self.assertTrue(os.environ['V32_RESUME_CHECKPOINT_PATH'].endswith('geometrize_gpt_pretrain_epoch_2.pt'))
            self.assertEqual(config.MODEL_D_MODEL,64)
            self.assertEqual(config.GPU_FP32_MASTER_WEIGHTS,False)

    def test_training_cli_does_not_reset_existing_run_silently(self):
        import train
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'checkpoints').mkdir();(root/'checkpoints/geometrize_gpt_pretrain_epoch_1.pt').write_bytes(b'old')
            args=train.parser().parse_args(['--data-dir',str(root),'--caption-root',str(root),'--checkpoint-dir',str(root/'checkpoints'),'--from-scratch'])
            with self.assertRaises(FileExistsError):train.configure(args)
if __name__=='__main__':unittest.main()
