from rrunet.unet_model import *

vis = True
vis = False
models=[
    Unet(3,1),
    Res_Unet(3,1),
    Att_Res_Unet(3,1),
    SE_Att_Res_Unet(3,1),
    CA_SE_Att_Res_Unet(3,1),
    Ringed_Res_Unet(3,1,vis),
    # SE_Ringed_Res_Unet(3,1),
    SE_Att_Ringed_Res_Unet(3,1),
    CA_SE_Att_Ringed_Res_Unet(3,1),
    DSE_Att_Ringed_Res_Unet(3,1),
    CA_Att_Res_Unet(3,1),
    CA_Att_Ringed_Res_Unet(3,1),
    SE_Unet(3,1),
    CA_Unet(3,1),
    CASE_Unet(3,1),
    CA_SE_SAtt_Ringed_Res_Unet(3,1),
    Att_Unet(3,1),
    SE_Ringed_Res_Unet(3,1),
    CA_SE_Fusion_Ringed_Res_Unet(3,1),
    Dual_CA_SE_Att_Ringed_Res_Unet(3,1),
    CA_SE_SAtt_Ringed_Res_Unet(3,1,[False,False,False,True]),
    CA_SE_SAtt_Ringed_Res_Unet(3,1,[False,False,True,True]),
    Double_CA_SE_Att_Ringed_Res_Unet(3,1),
    Fusion_Ringed_Res_Unet(3,1),
    CA_SE_Fusion_Ringed_Res_Unet(3,1),
    Dual_CA_SE_Fusion_Ringed_Res_Unet(3,1,vis),
    Dual_CA_SE_AFusion_Ringed_Res_Unet(3,1,vis), # 25
    Dual_CA_SE_DAFusion_Ringed_Res_Unet(3,1,vis),
    Dual_CA_SE_ResAFusion_Ringed_Res_Unet(3,1,vis),
    Dual_CA_SE_AFusion_Ringed_Res_Unet2(3,1,vis),
    Dual_CA_SE_AFusion_Ringed_Res_Unet4(3,1,vis),
    Dual_CA_SE_AFusionR_Ringed_Res_Unet(3,1,vis) # 30
]