"""Center-coordinate legal sets matching renderer WGSL clamp(start,end-1)."""
import torch
from token_layout import TOKEN_LAYOUT,XY_BINS_PER_PIXEL

REGION_LEGAL_VERSION='renderer_center_bounds_v1'


def center_bin_bounds(stage,region_id=None,canvas_size=512):
    if canvas_size%8:raise ValueError('Canvas must be divisible by8')
    scale=XY_BINS_PER_PIXEL
    if stage=='global':return (0,(canvas_size-1)*scale,0,(canvas_size-1)*scale)
    if stage not in ('coarse','fine'):raise ValueError('Unknown region stage')
    side=4 if stage=='coarse' else 8
    if region_id is None or not 0<=int(region_id)<side*side:raise ValueError('Invalid region ID')
    row,col=divmod(int(region_id),side);size=canvas_size//side
    return (col*size*scale,((col+1)*size-1)*scale,
            row*size*scale,((row+1)*size-1)*scale)


class RegionCandidateMasks:
    def __init__(self,device,canvas_size=512):
        bounds=[center_bin_bounds('global',canvas_size=canvas_size)]
        bounds += [center_bin_bounds('coarse',i,canvas_size) for i in range(16)]
        bounds += [center_bin_bounds('fine',i,canvas_size) for i in range(64)]
        limits=torch.tensor(bounds,device=device)
        x=torch.arange(TOKEN_LAYOUT.x_bins,device=device)[None]
        y=torch.arange(TOKEN_LAYOUT.y_bins,device=device)[None]
        self.x=(x>=limits[:,0:1])&(x<=limits[:,1:2])
        self.y=(y>=limits[:,2:3])&(y<=limits[:,3:4])
        self.device=device
    def for_lanes(self,stages,regions):
        codes=[0 if s=='global' or r is None else (1+int(r) if s=='coarse' else 17+int(r))
               for s,r in zip(stages,regions)]
        index=torch.tensor(codes,device=self.device,dtype=torch.long)
        return self.x.index_select(0,index),self.y.index_select(0,index)
