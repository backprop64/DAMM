from DAMM.tracking import PromptableVideoTracker

sam_config = 'sam2_hiera_l.yaml'
sam_checkpoint = '/nfs/turbo/lsa-adae/kaulg/datasets/DAMM/models/sam2_hiera_large.pt'
damm_config = '/nfs/turbo/lsa-adae/kaulg/datasets/DAMM/models/DAMM_config.yaml'
damm_checkpoint = '/nfs/turbo/lsa-adae/kaulg/datasets/DAMM/models/DAMM_weights.pth'

mouse_tracker = PromptableVideoTracker(sam_config,
                                         sam_checkpoint,
                                         damm_config,
                                         damm_checkpoint)