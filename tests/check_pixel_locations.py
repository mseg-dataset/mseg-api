

import imageio
import matplotlib.pyplot as plt
from pathlib import Path


# given as [y,x]
# fpaths = [
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_ade20k-150-relabeled/ade20k-150-relabeled_counter-other_ADE_val_00001477_157.png', (160,234)], # (153,271)
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_ade20k-150-relabeled/ade20k-150-relabeled_wine glass_ADE_val_00001949_134.png', (300,478)],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_ade20k-150-relabeled/ade20k-150-relabeled_teddy_bear_ADE_train_00001951_185.png', (328,508)],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_ade20k-150-relabeled/ade20k-150-relabeled_bicyclist_ADE_train_00017404_215.png', (268, 373)],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_ade20k-150-relabeled/ade20k-150-relabeled_bicyclist_ADE_train_00017384_120.png', (398, 55)],
#	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_ade20k-150-relabeled/ade20k-150-relabeled_zebra_ADE_train_00016125_113.png', (504,200)],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_ade20k-150-relabeled/ade20k-150-relabeled_nightstand_ADE_train_00004036_119.png', (195, 37)],
# ]

# fpaths = [

# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_cityscapes-19-relabeled/cityscapes-19-relabeled_person-nonrider_seqfrankfurt_frankfurt_000001_038844_leftImg8bit_82.jpg', (341,649) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_cityscapes-19-relabeled/cityscapes-19-relabeled_rider-other_seqaachen_aachen_000014_000019_leftImg8bit_84.jpg', (542,531) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_cityscapes-19-relabeled/cityscapes-19-relabeled_motorcyclist_seqbremen_bremen_000052_000019_leftImg8bit_102.jpg', (403, 1048) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_cityscapes-19-relabeled/cityscapes-19-relabeled_bicyclist_seqhamburg_hamburg_000000_068916_leftImg8bit_223.jpg', (353, 1181) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_cityscapes-19-relabeled/cityscapes-19-relabeled_bicycle_seqtubingen_tubingen_000060_000019_leftImg8bit_57.jpg', (474, 760) ]

# ]

# fpaths = [
# 	[ '/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_bdd-relabeled/bdd-relabeled_person-nonrider_a706da19-14468d02_11.jpg', (300,1248) ],
# 	[ '/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_bdd-relabeled/bdd-relabeled_motorcyclist_a91b7555-00000590_12.jpg', (470, 888) ],
# 	[ '/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_bdd-relabeled/bdd-relabeled_bicyclist_1ff92f74-697a077e_11.jpg', (270, 946) ],
# 	[ '/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_bdd-relabeled/bdd-relabeled_person-nonrider_8413f861-580b500d_12.jpg', (416,926) ],
# 	[ '/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_bdd-relabeled/bdd-relabeled_motorcyclist_972ab49a-6a6eeaf5_11.jpg', (396, 284) ],
# 	[ '/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_bdd-relabeled/bdd-relabeled_bicyclist_3516379e-43f6a6ba_12.jpg', (410, 654) ],
# 	[ '/Users/johnlamb/Downloads/bdd100k/seg/images/train/3516379e-43f6a6ba.jpg', (410, 654) ]
# ]

# fpaths = [
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_coco-panoptic-133-relabeled/coco-panoptic-133-relabeled_motorcyclist_000000459634_6904408.png', (299,99) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_coco-panoptic-133-relabeled/coco-panoptic-133-relabeled_kitchen island_000000438774_5403260.png', (199, 341) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_coco-panoptic-133-relabeled/coco-panoptic-133-relabeled_bathroom counter_000000376310_2631210.png', (418, 558) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_coco-panoptic-133-relabeled/coco-panoptic-133-relabeled_bicyclist_000000067208_4211027.png', (83, 404) ],
#	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_coco-panoptic-133-relabeled/coco-panoptic-133-relabeled_seat_000000016509_9079654.png', (46, 133) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_coco-panoptic-133-relabeled/coco-panoptic-133-relabeled_wheelchair_000000091349_3093585.png', (245, 396) ],
# ]

# fpaths = [
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_idd-39-relabeled/idd-39-relabeled_rider-other_seq148_000690_leftImg8bit_67.jpg', (591, 1013) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_idd-39-relabeled/idd-39-relabeled_bicyclist_seq181_016419_leftImg8bit_41.jpg', (365, 889) ],
# 	# ['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_idd-39-relabeled/idd-39-relabeled_rider-other_seq48_601929_leftImg8bit_79.jpg', ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_idd-39-relabeled/idd-39-relabeled_person-non-rider_seq1_148425_leftImg8bit_110.jpg', (458, 822) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_idd-39-relabeled/idd-39-relabeled_motorcyclist_seq98_776293_leftImg8bit_110.jpg', (502, 1101) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_idd-39-relabeled/idd-39-relabeled_box_seq173_316862_leftImg8bit_34.jpg', (752, 761) ], 
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_idd-39-relabeled/idd-39-relabeled_bicyclist_seq158_867833_leftImg8bit_54.jpg', (510, 453) ], 
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_idd-39-relabeled/idd-39-relabeled_backpack_seq173_318463_leftImg8bit_50.jpg', (912, 1191) ],
# ]

# fpaths = [
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_mapillary-public65-relabeled/mapillary-public65-relabeled_dog_mapillary_OkOJlU8l98KqPEE05dv1tA_256.jpg', (1773, 886) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_mapillary-public65-relabeled/mapillary-public65-relabeled_sea_mapillary_EZJTbIpLNOfHnw3DvA1NEA_7936.jpg', (1921, 3000) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_mapillary-public65-relabeled/mapillary-public65-relabeled_horse_mapillary_CH1TcYj0ki_kw3Yc3oOiZA_258.jpg', (1724, 180) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_mapillary-public65-relabeled/mapillary-public65-relabeled_fountain_mapillary_1fhI-qv6dVR-1Kc8Oafb3Q_7936.jpg', (1059, 2663) ]
# ]

fpaths = [
	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_sunrgbd-37-relabeled/sunrgbd-37-relabeled_bathroom counter_img-002164_11.jpg', (350, 483) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_sunrgbd-37-relabeled/sunrgbd-37-relabeled_sconce_img-004159_34.jpg', (204, 284) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_sunrgbd-37-relabeled/sunrgbd-37-relabeled_door_img-004239_4.jpg', (190,30) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_sunrgbd-37-relabeled/sunrgbd-37-relabeled_desk_img-004930_11.jpg', (429, 143) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_sunrgbd-37-relabeled/sunrgbd-37-relabeled_chandelier_img-001692_34.jpg', (3,459) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_sunrgbd-37-relabeled/sunrgbd-37-relabeled_bathroom counter_img-002347_11.jpg', (254,34) ],
# 	['/Users/johnlamb/Downloads/rewrote_ade20k_check_validity/target_visual_check_sunrgbd-37-relabeled/sunrgbd-37-relabeled_swivel chair_img-000253_4.jpg', (303, 463) ],
]

for (fpath,coords) in fpaths:
	# for fpath in fpaths:
	img = imageio.imread(fpath)
	plt.imshow(img)
	plt.title(Path(fpath).name)
	y,x = coords
	plt.scatter(x,y,20,color='r',marker='+')
	plt.show()


fpaths = [
	#['/Users/johnlamb/Documents/bugs/ADE_train_00016125.jpg','/Users/johnlamb/Documents/bugs/ADE_train_00016125.png', (504,200) ],
	#['/Users/johnlamb/Documents/bugs/000000016509.jpg','/Users/johnlamb/Documents/bugs/000000016509.png', (46, 133) ],
	['/Users/johnlamb/Documents/bugs/img-002164.jpg','/Users/johnlamb/Documents/bugs/00002164.png', (350, 483) ],
]


for (rgb_fpath, label_fpath, coords) in fpaths:
	# for fpath in fpaths:
	img = imageio.imread(label_fpath)
	plt.imshow(img)
	plt.title(Path(rgb_fpath).name)
	y,x = coords
	plt.scatter(x,y,20,color='r',marker='+')
	plt.show()







