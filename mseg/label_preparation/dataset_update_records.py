#!/usr/bin/python3



from mseg.label_preparation.relabeled_data_containers import DatasetClassUpdateRecord


""" COCO Panoptic"""
cocop_update_records = [
	# cabinet-merged-classification,fence-guardrail-classification
	# person_rider_classification,rug-merged,showercurtain_curtainother_classification

							# dataset        # split   # orig u name    # new u name    # path to txt file
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'cabinet', 		'desk',				'cocop_cabinet-merged/cocopanoptic_desk_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'cabinet', 		'bookshelf',		'cocop_cabinet-merged/cocopanoptic_bookshelf_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'cabinet', 		'bookshelf',		'cocop_cabinet-merged/cocopanoptic_bookshelf_val.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'cabinet', 		'cabinet',			'cocop_cabinet-merged/cocopanoptic_cabinet_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'cabinet', 		'cabinet',			'cocop_cabinet-merged/cocopanoptic_cabinet_val.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'cabinet', 		'unlabeled',		'cocop_cabinet-merged/cocopanoptic_cabinetmerged_unlabel_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'cabinet', 		'unlabeled',		'cocop_cabinet-merged/cocopanoptic_cabinetmerged_unlabel_val.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'cabinet', 		'counter_other',  	'cocop_cabinet-merged/cocopanoptic_counter_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'cabinet', 		'chest_of_drawers',	'cocop_cabinet-merged/cocopanoptic_dresser_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'cabinet', 		'chest_of_drawers',	'cocop_cabinet-merged/cocopanoptic_dresser_val.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'cabinet', 		'nightstand',		'cocop_cabinet-merged/cocopanoptic_nightstand_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'cabinet', 		'nightstand',		'cocop_cabinet-merged/cocopanoptic_nightstand_val.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'cabinet', 		'wardrobe',			'cocop_cabinet-merged/cocopanoptic_wardrobe_train.txt'),

	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'fence', 		'fence',   		'cocop_fence-guardrail/cocopanoptic_fence_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'fence', 		'fence',     	'cocop_fence-guardrail/cocopanoptic_fence_val.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'fence', 		'guard_rail',	'cocop_fence-guardrail/cocopanoptic_guardrail_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'fence', 		'guard_rail',	'cocop_fence-guardrail/cocopanoptic_guardrail_val.txt'),

	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'rug_floormat', 	'unlabeled', 		'cocop_rug-merged/cocopanoptic_rug-merged_unlabel-list_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'rug_floormat', 	'unlabeled',   		'cocop_rug-merged/cocopanoptic_rug-merged_unlabel-list_val.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'rug_floormat', 	'floor',     		'cocop_rug-merged/floor-carpet_80percent_confidence_cocopanoptic_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'rug_floormat', 	'floor',       		'cocop_rug-merged/floor-carpet_80percent_confidence_cocopanoptic_val.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'rug_floormat', 	'rug_floormat',		'cocop_rug-merged/rug-floormat_80percent_confidence_cocopanoptic_train.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'rug_floormat', 	'rug_floormat',  	'cocop_rug-merged/rug-floormat_80percent_confidence_cocopanoptic_val.txt'),

	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'curtain_other', 'curtain_other',   	'cocop_curtain/2019_08_16_coco_other-curtain_80percent_conf_valsplit.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'val', 	'curtain_other', 'shower_curtain',  	'cocop_curtain/2019_08_16_coco_showercurtain_80percent_conf_valsplit.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'curtain_other', 'curtain_other', 		'cocop_curtain/2019_08_17_coco_other-curtain_train_80percent_conf.txt'),
	DatasetClassUpdateRecord('cocopanoptic', 'train', 	'curtain_other', 'shower_curtain',		'cocop_curtain/2019_08_17_coco_showercurtain_train_80percent_conf.txt'),

	# cocop_cocop_chair (COCOP CHAIR-> UNIVERSAL CHAIR-OTHER )
							# dataset  			# split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'armchair', 		'cocop_chair/cocop_train_cocop_chair_to_armchair.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'basket', 			'cocop_chair/cocop_train_cocop_chair_to_basket.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'bench', 			'cocop_chair/cocop_train_cocop_chair_to_bench.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'chair_other', 		'cocop_chair/cocop_train_cocop_chair_to_chair-other.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'unlabeled', 		'cocop_chair/cocop_train_cocop_chair_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'ottoman', 			'cocop_chair/cocop_train_cocop_chair_to_ottoman.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'seat', 			'cocop_chair/cocop_train_cocop_chair_to_seat.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 'slow_wheeled_object', 	'cocop_chair/cocop_train_cocop_chair_to_slowwheeledobject.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'stool', 			'cocop_chair/cocop_train_cocop_chair_to_stool.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'chair_other', 	'swivel_chair', 	'cocop_chair/cocop_train_cocop_chair_to_swivel-chair.txt'),

	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 	'chair_other', 'armchair', 				'cocop_chair/cocop_val_cocop_chair_to_armchair.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 	'chair_other', 'chair_other', 			'cocop_chair/cocop_val_cocop_chair_to_chair-other.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 	'chair_other', 'unlabeled', 			'cocop_chair/cocop_val_cocop_chair_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 	'chair_other', 'ottoman', 				'cocop_chair/cocop_val_cocop_chair_to_ottoman.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 	'chair_other', 'seat', 					'cocop_chair/cocop_val_cocop_chair_to_seat.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 	'chair_other', 'slow_wheeled_object', 	'cocop_chair/cocop_val_cocop_chair_to_slowwheeledobject.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 	'chair_other', 'stool', 				'cocop_chair/cocop_val_cocop_chair_to_stool.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 	'chair_other', 'swivel_chair', 			'cocop_chair/cocop_val_cocop_chair_to_swivel-chair.txt'),

	# cocop_cocop_counter (COCO COUNTER -> UNIVERSAL COUNTER-OTHER)
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'counter_other', 'bathroom_counter',	'cocop_cocop_counter/cocop_train_cocop_counter_to_bathroom-counter.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'counter_other', 'counter_other',		'cocop_cocop_counter/cocop_train_cocop_counter_to_counter-other.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'counter_other', 'kitchen_island',		'cocop_cocop_counter/cocop_train_cocop_counter_to_kitchen-island.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'counter_other', 'nightstand',			'cocop_cocop_counter/cocop_train_cocop_counter_to_nightstand.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'counter_other', 'unlabeled',			'cocop_cocop_counter/cocop_train_cocop_counter_to_None_of_these.txt'),

	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'counter_other', 'bathroom_counter',	'cocop_cocop_counter/cocop_val_cocop_counter_to_bathroom-counter.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'counter_other', 'counter_other',		'cocop_cocop_counter/cocop_val_cocop_counter_to_counter-other.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'counter_other', 'kitchen_island',		'cocop_cocop_counter/cocop_val_cocop_counter_to_kitchen-island.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'counter_other', 'unlabeled',			'cocop_cocop_counter/cocop_val_cocop_counter_to_None_of_these.txt'),

	# cocop_cocop_diningtable (COCOP {table-merged, dining table} -> UNIVERSAL TABLE )
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'bathroom_counter', 	'cocop_diningtable/cocop_train_cocop_diningtable_to_bathroom-counter.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'box', 				'cocop_diningtable/cocop_train_cocop_diningtable_to_box.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'chest_of_drawers', 	'cocop_diningtable/cocop_train_cocop_diningtable_to_chestofdrawers.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'counter_other', 		'cocop_diningtable/cocop_train_cocop_diningtable_to_counter-other.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'desk', 				'cocop_diningtable/cocop_train_cocop_diningtable_to_desk-and-table.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'desk', 				'cocop_diningtable/cocop_train_cocop_diningtable_to_desk.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'kitchen_island', 	'cocop_diningtable/cocop_train_cocop_diningtable_to_kitchen-island.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'nightstand', 		'cocop_diningtable/cocop_train_cocop_diningtable_to_nightstand.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'unlabeled', 			'cocop_diningtable/cocop_train_cocop_diningtable_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'table', 'table', 				'cocop_diningtable/cocop_train_cocop_diningtable_to_table.txt'),

	DatasetClassUpdateRecord('cocopanoptic-v1', 'val',	 'table', 'box', 				'cocop_diningtable/cocop_val_diningtable_to_box.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val',	 'table', 'counter_other', 		'cocop_diningtable/cocop_val_diningtable_to_counterother.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val',	 'table', 'desk', 				'cocop_diningtable/cocop_val_diningtable_to_desk.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val',	 'table', 'desk', 				'cocop_diningtable/cocop_val_diningtable_to_deskandtable.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val',	 'table', 'kitchen_island', 	'cocop_diningtable/cocop_val_diningtable_to_kitchenisland.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val',	 'table', 'table', 				'cocop_diningtable/cocop_val_diningtable_to_table.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val',	 'table', 'unlabeled', 			'cocop_diningtable/cocop_val_diningtable_to_unlabel.txt'),

	# cocop_cocop_table-merged
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'base', 			'cocop_table-merged/cocop_train_cocop_table-merged_to_base.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'bathroom_counter','cocop_table-merged/cocop_train_cocop_table-merged_to_bathroom-counter.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'cabinet', 		'cocop_table-merged/cocop_train_cocop_table-merged_to_cabinet.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'chest_of_drawers','cocop_table-merged/cocop_train_cocop_table-merged_to_chestofdrawers.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'counter_other', 	'cocop_table-merged/cocop_train_cocop_table-merged_to_counter-other.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'desk', 			'cocop_table-merged/cocop_train_cocop_table-merged_to_desk.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'kitchen_island', 	'cocop_table-merged/cocop_train_cocop_table-merged_to_kitchen-island.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'laptop', 			'cocop_table-merged/cocop_train_cocop_table-merged_to_laptop.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'mountain_hill', 	'cocop_table-merged/cocop_train_cocop_table-merged_to_mountainhill.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'nightstand', 		'cocop_table-merged/cocop_train_cocop_table-merged_to_nightstand.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'unlabeled', 		'cocop_table-merged/cocop_train_cocop_table-merged_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'pool_table', 		'cocop_table-merged/cocop_train_cocop_table-merged_to_pooltable.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'stool', 			'cocop_table-merged/cocop_train_cocop_table-merged_to_stool.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'desk', 			'cocop_table-merged/cocop_train_cocop_table-merged_to_table-used-as-desk.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'table', 'table', 			'cocop_table-merged/cocop_train_cocop_table-merged_to_table.txt'),

	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'table', 'bathroom_counter','cocop_table-merged/cocop_val_cocop_table-merged_to_bathroomcounter.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'table', 'counter_other', 	'cocop_table-merged/cocop_val_cocop_table-merged_to_counterother.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'table', 'desk', 			'cocop_table-merged/cocop_val_cocop_table-merged_to_desk.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'table', 'kitchen_island', 	'cocop_table-merged/cocop_val_cocop_table-merged_to_kitchenisland.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'table', 'nightstand', 		'cocop_table-merged/cocop_val_cocop_table-merged_to_nightstand.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'table', 'desk', 			'cocop_table-merged/cocop_val_cocop_table-merged_to_table-used-as-desk.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'table', 'table', 			'cocop_table-merged/cocop_val_cocop_table-merged_to_table.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'table', 'unlabeled', 		'cocop_table-merged/cocop_val_cocop_table-merged_to_unlabel.txt'),


	# cocop_cocop_light (COCO LIGHT-> UNIVERSAL LIGHT-OTHER)
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train',	'light_other', 'chandelier', 	'cocop_light/cocop_train_cocop_light_to_chandelier.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train',	'light_other', 'lamp', 			'cocop_light/cocop_train_cocop_light_to_lamp.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train',	'light_other', 'light_other', 	'cocop_light/cocop_train_cocop_light_to_light-other.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train',	'light_other', 'unlabeled', 	'cocop_light/cocop_train_cocop_light_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train',	'light_other', 'sconce', 		'cocop_light/cocop_train_cocop_light_to_sconce.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train',	'light_other', 'streetlight', 	'cocop_light/cocop_train_cocop_light_to_streetlight.txt'),

	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'light_other', 'chandelier', 	'cocop_light/cocop_val_cocop_light_to_chandelier.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'light_other', 'lamp', 			'cocop_light/cocop_val_cocop_light_to_lamp.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'light_other', 'light_other', 	'cocop_light/cocop_val_cocop_light_to_light-other.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'light_other', 'unlabeled', 	'cocop_light/cocop_val_cocop_light_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'light_other', 'sconce', 		'cocop_light/cocop_val_cocop_light_to_sconce.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'light_other', 'streetlight', 	'cocop_light/cocop_val_cocop_light_to_streetlight.txt'),


	# cocopanoptic_cocop_motorcyclist_bicyclist (COCO PERSON -> UNIVERSAL PERSON-NON-RIDER)
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'person', 'bicyclist', 			'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_train_cocop_motorcyclist_bicyclist_to_bicyclist.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'person', 'motorcyclist', 		'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_train_cocop_motorcyclist_bicyclist_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'person', 'unlabeled', 			'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_train_cocop_motorcyclist_bicyclist_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'person', 'person', 	'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_train_cocop_motorcyclist_bicyclist_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 'person', 'rider_other', 		'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_train_cocop_motorcyclist_bicyclist_to_rider-other.txt'),

	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 'person', 	'bicyclist',			'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_val_cocop_motorcyclist_bicyclist_to_bicyclist.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 'person', 	'motorcyclist',			'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_val_cocop_motorcyclist_bicyclist_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 'person', 	'person',		'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_val_cocop_motorcyclist_bicyclist_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 'person', 	'rider_other',			'cocopanoptic_cocop_motorcyclist_bicyclist/cocopanoptic_val_cocop_motorcyclist_bicyclist_to_rider-other.txt'),

	# cocopanoptic_cocop_nonrider
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'person', 'bicyclist', 		'cocopanoptic_cocop_nonrider/cocopanoptic_train_cocop_nonrider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'person', 'motorcyclist', 	'cocopanoptic_cocop_nonrider/cocopanoptic_train_cocop_nonrider_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'person', 'unlabeled', 		'cocopanoptic_cocop_nonrider/cocopanoptic_train_cocop_nonrider_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'person', 'person', 'cocopanoptic_cocop_nonrider/cocopanoptic_train_cocop_nonrider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'train', 	'person', 'rider_other', 		'cocopanoptic_cocop_nonrider/cocopanoptic_train_cocop_nonrider_to_rider-other.txt'),

	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'person', 'bicyclist', 		'cocopanoptic_cocop_nonrider/cocopanoptic_val_cocop_nonrider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'person', 'motorcyclist', 	'cocopanoptic_cocop_nonrider/cocopanoptic_val_cocop_nonrider_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'person', 'unlabeled', 		'cocopanoptic_cocop_nonrider/cocopanoptic_val_cocop_nonrider_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'person', 'person', 'cocopanoptic_cocop_nonrider/cocopanoptic_val_cocop_nonrider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('cocopanoptic-v1', 'val', 		'person', 'rider_other', 		'cocopanoptic_cocop_nonrider/cocopanoptic_val_cocop_nonrider_to_rider-other.txt'),

	# Phase 3, cocop_waterother
							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'fountain', 		'cocop_waterother/cocop_train_cocop_waterother_to_fountain.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'unlabeled', 		'cocop_waterother/cocop_train_cocop_waterother_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'playingfield', 	'cocop_waterother/cocop_train_cocop_waterother_to_playingfield.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'river_lake', 	'cocop_waterother/cocop_train_cocop_waterother_to_river-lake.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'sea', 			'cocop_waterother/cocop_train_cocop_waterother_to_sea.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'swimming_pool', 	'cocop_waterother/cocop_train_cocop_waterother_to_swimming-pool.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'terrain', 		'cocop_waterother/cocop_train_cocop_waterother_to_terrain.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'wall', 			'cocop_waterother/cocop_train_cocop_waterother_to_wall.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'water_other', 	'cocop_waterother/cocop_train_cocop_waterother_to_water-other.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'waterfall', 		'cocop_waterother/cocop_train_cocop_waterother_to_waterfall.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 'water_other', 'window', 		'cocop_waterother/cocop_train_cocop_waterother_to_window.txt'),

	# Phase 3,cocop_runway_pavement
							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('cocop','train', 'sidewalk_pavement', 'runway',			'cocop_runway_pavement/cocop_train_pavementmerged_to_airport_runway.txt'),
	DatasetClassUpdateRecord('cocop','train', 'sidewalk_pavement', 'floor',	 			'cocop_runway_pavement/cocop_train_pavementmerged_to_floor.txt'),
	DatasetClassUpdateRecord('cocop','train', 'sidewalk_pavement', 'sidewalk_pavement',	'cocop_runway_pavement/cocop_train_pavementmerged_to_pavementmerged.txt'),
	DatasetClassUpdateRecord('cocop','train', 'sidewalk_pavement', 'pier_wharf',	 	'cocop_runway_pavement/cocop_train_pavementmerged_to_pierwharf.txt'),

	DatasetClassUpdateRecord('cocop','val' 	, 'sidewalk_pavement', 'runway',			'cocop_runway_pavement/cocop_val_pavementmerged_to_airport_runway.txt'),
	DatasetClassUpdateRecord('cocop','val' 	, 'sidewalk_pavement', 'sidewalk_pavement',	'cocop_runway_pavement/cocop_val_pavementmerged_to_pavementmerged.txt'),

	# Phase 3,cocop_runway_road
							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('cocop', 'train', 	'road', 		'runway', 			'cocop_runway_road/cocop_train_road_to_airportrunway.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'road', 		'sidewalk_pavement','cocop_runway_road/cocop_train_road_to_pavementsidewalk.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'road', 		'road', 			'cocop_runway_road/cocop_train_road_to_road.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'road', 		'unlabeled', 		'cocop_runway_road/cocop_train_road_to_unlabel.txt'),

	DatasetClassUpdateRecord('cocop', 'val', 	'road', 		'runway', 			'cocop_runway_road/cocop_val_road_to_airportrunway.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'road', 		'sidewalk_pavement','cocop_runway_road/cocop_val_road_to_pavementsidewalk.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'road', 		'road', 			'cocop_runway_road/cocop_val_road_to_road.txt'),


	# Phase 3,cocop_cocop_platform
							# dataset  # split  # orig u name  	# new u name    # path to txt file
	DatasetClassUpdateRecord('cocop', 'train', 	'platform', 	'bridge', 		'cocop_platform/cocop_train_cocop_platform_to_bridge.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'platform', 	'unlabeled', 	'cocop_platform/cocop_train_cocop_platform_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'platform', 	'pier_wharf', 	'cocop_platform/cocop_train_cocop_platform_to_pier-wharf.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'platform', 	'platform', 	'cocop_platform/cocop_train_cocop_platform_to_platform.txt'),

	DatasetClassUpdateRecord('cocop', 'val', 	'platform', 	'building', 	'cocop_platform/cocop_val_cocop_platform_to_grandstand.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'platform', 	'unlabeled', 	'cocop_platform/cocop_val_cocop_platform_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'platform', 	'pier_wharf', 	'cocop_platform/cocop_val_cocop_platform_to_pier-wharf.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'platform', 	'platform', 	'cocop_platform/cocop_val_cocop_platform_to_platform.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'platform', 	'playingfield', 'cocop_platform/cocop_val_cocop_platform_to_playingfield.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'platform', 	'stage', 		'cocop_platform/cocop_val_cocop_platform_to_stage.txt'),

	# Phase 3, cocop_cocop_bridge
							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'bridge',	 	'cocop_bridge/cocop_train_cocop_bridge_to_bridge.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'building',		'cocop_bridge/cocop_train_cocop_bridge_to_building.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'building',		'cocop_bridge/cocop_train_cocop_bridge_to_grandstand.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'building',	 	'cocop_bridge/cocop_train_cocop_bridge_to_house.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'unlabeled',	'cocop_bridge/cocop_train_cocop_bridge_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'pier_wharf',	'cocop_bridge/cocop_train_cocop_bridge_to_pier-wharf.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'platform',		'cocop_bridge/cocop_train_cocop_bridge_to_platform.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'runway',	 	'cocop_bridge/cocop_train_cocop_bridge_to_runway.txt'),
	DatasetClassUpdateRecord('cocop', 'train', 	'bridge', 		'vegetation',	'cocop_bridge/cocop_train_cocop_bridge_to_vegetation.txt'),

	DatasetClassUpdateRecord('cocop', 'val', 	'bridge', 		'bridge', 		'cocop_bridge/cocop_val_cocop_bridge_to_bridge.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'bridge', 		'building', 	'cocop_bridge/cocop_val_cocop_bridge_to_building.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'bridge', 		'building', 	'cocop_bridge/cocop_val_cocop_bridge_to_house.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'bridge', 		'unlabeled',	'cocop_bridge/cocop_val_cocop_bridge_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocop', 'val', 	'bridge', 		'pier_wharf', 	'cocop_bridge/cocop_val_cocop_bridge_to_pier-wharf.txt'),

							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'awning', 		'cocop_tent/cocop_train_cocop_tent_to_awning.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'base', 		'cocop_tent/cocop_train_cocop_tent_to_base.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'building', 	'cocop_tent/cocop_train_cocop_tent_to_building.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'fence', 		'cocop_tent/cocop_train_cocop_tent_to_fence.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'unlabeled', 	'cocop_tent/cocop_train_cocop_tent_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'billboard', 	'cocop_tent/cocop_train_cocop_tent_to_signboard.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'tent', 		'cocop_tent/cocop_train_cocop_tent_to_tent.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'umbrella', 	'cocop_tent/cocop_train_cocop_tent_to_umbrella.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'vegetation',	'cocop_tent/cocop_train_cocop_tent_to_vegetation.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'wall', 		'cocop_tent/cocop_train_cocop_tent_to_wall.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'window', 		'cocop_tent/cocop_train_cocop_tent_to_window.txt'),
	DatasetClassUpdateRecord('cocop', 	'train', 	'tent', 	'kite',			'cocop_tent/cocop_train_cocop_tent_to_windsurfingkite.txt'),

							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('cocop', 	'val', 		'tent', 	'awning',		'cocop_tent/cocop_val_cocop_tent_to_awning.txt'),
	DatasetClassUpdateRecord('cocop', 	'val', 		'tent', 	'building',		'cocop_tent/cocop_val_cocop_tent_to_building.txt'),
	DatasetClassUpdateRecord('cocop', 	'val', 		'tent', 	'unlabeled',	'cocop_tent/cocop_val_cocop_tent_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cocop', 	'val', 		'tent', 	'tent',			'cocop_tent/cocop_val_cocop_tent_to_tent.txt'),
	DatasetClassUpdateRecord('cocop', 	'val', 		'tent', 	'umbrella',		'cocop_tent/cocop_val_cocop_tent_to_umbrella.txt'),
	DatasetClassUpdateRecord('cocop', 	'val', 		'tent', 	'window',		'cocop_tent/cocop_val_cocop_tent_to_window.txt')

]



""" ADE20K """
ade20k_update_records = [

	# Phase1 - fence guardrail, person-rider, shower-curtain, curtain-other, snow-mountain-hill-ade20k
							# dataset     # split   # orig u name    # new u name # path to txt file
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'fence', 		'fence', 		'ade20k_fence_guardrail/ade20k_fence_train.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'fence', 		'fence', 		'ade20k_fence_guardrail/ade20k_fence_val.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'fence', 		'guard_rail', 	'ade20k_fence_guardrail/ade20k_guardrail_train.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'fence', 		'guard_rail', 	'ade20k_fence_guardrail/ade20k_guardrail_val.txt'),

	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'curtain_other', 'curtain_other', 	'ade20k_curtain/2019_10_31_ade20k_other-curtain_valsplit.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'curtain_other', 'shower_curtain', 	'ade20k_curtain/2019_10_31_ade20k_showercurtain_valsplit.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'curtain_other', 'curtain_other', 	'ade20k_curtain/2019_11_1_ade20k_other-curtain_trainsplit.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'curtain_other', 'shower_curtain', 	'ade20k_curtain/2019_11_1_ade20k_showercurtain_trainsplit.txt'),

	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'mountain_hill', 'mountain_hill', 	'ade20k_snow-mountain-hill/ade20k_mountainhill_train.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'mountain_hill', 'mountain_hill', 	'ade20k_snow-mountain-hill/ade20k_mountainhill_val.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'mountain_hill', 'snow', 			'ade20k_snow-mountain-hill/ade20k_snow_train.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'mountain_hill', 'snow', 			'ade20k_snow-mountain-hill/ade20k_snow_val.txt'),

	# ADE20K_ade20k_animal
							# dataset  # split  	# orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'animal_other',	'ade20k_animal/ADE20K_train_ade20k_animal_to_animal-other.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'bear',	 		'ade20k_animal/ADE20K_train_ade20k_animal_to_bear.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'bird',	 		'ade20k_animal/ADE20K_train_ade20k_animal_to_bird.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'cat',	 		'ade20k_animal/ADE20K_train_ade20k_animal_to_cat.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'cow',	 		'ade20k_animal/ADE20K_train_ade20k_animal_to_cow.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'dog',	 		'ade20k_animal/ADE20K_train_ade20k_animal_to_dog.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'elephant',	 	'ade20k_animal/ADE20K_train_ade20k_animal_to_elephant.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'giraffe',	 	'ade20k_animal/ADE20K_train_ade20k_animal_to_giraffe.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'horse',	 		'ade20k_animal/ADE20K_train_ade20k_animal_to_horse.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'unlabeled',		'ade20k_animal/ADE20K_train_ade20k_animal_to_None_of_these.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other','plaything_other','ade20k_animal/ADE20K_train_ade20k_animal_to_plaything.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'sheep',	 		'ade20k_animal/ADE20K_train_ade20k_animal_to_sheep.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'animal_other', 'zebra',	 		'ade20k_animal/ADE20K_train_ade20k_animal_to_zebra.txt'),

	DatasetClassUpdateRecord('ade20k-v1', 'val',	'animal_other', 'animal_other', 'ade20k_animal/ADE20K_val_ade20k_animal_to_animal-other.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'animal_other', 'bear',			'ade20k_animal/ADE20K_val_ade20k_animal_to_bear.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'animal_other', 'bird',			'ade20k_animal/ADE20K_val_ade20k_animal_to_bird.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'animal_other', 'cow',			'ade20k_animal/ADE20K_val_ade20k_animal_to_cow.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'animal_other', 'elephant',		'ade20k_animal/ADE20K_val_ade20k_animal_to_elephant.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'animal_other', 'horse',		'ade20k_animal/ADE20K_val_ade20k_animal_to_horse.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'animal_other', 'sheep',		'ade20k_animal/ADE20K_val_ade20k_animal_to_sheep.txt'),

	# ade20k_ade20k_chest_of_drawers
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'chest_of_drawers', 'cabinet',			'ade20k_chest_of_drawers/ade20k_train_ade20k_chest_of_drawers_to_cabinet.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'chest_of_drawers', 'chest_of_drawers',	'ade20k_chest_of_drawers/ade20k_train_ade20k_chest_of_drawers_to_chest-of-drawers.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'chest_of_drawers', 'nightstand',		'ade20k_chest_of_drawers/ade20k_train_ade20k_chest_of_drawers_to_nightstand.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 'chest_of_drawers', 'unlabeled',			'ade20k_chest_of_drawers/ade20k_train_ade20k_chest_of_drawers_to_None_of_these.txt'),

	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'chest_of_drawers',	'cabinet',			'ade20k_chest_of_drawers/ade20k_val_ade20k_chest_of_drawers_to_cabinet.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'chest_of_drawers',	'chest_of_drawers',	'ade20k_chest_of_drawers/ade20k_val_ade20k_chest_of_drawers_to_chest-of-drawers.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'chest_of_drawers',	'desk',				'ade20k_chest_of_drawers/ade20k_val_ade20k_chest_of_drawers_to_desk.txt'),

	# TODO
	# ADE20K_ade20k_food
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'food_other', 	'cake',			'ade20k_food/ADE20K_train_ade20k_food_to_cake.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'food_other', 	'food_other',	'ade20k_food/ADE20K_train_ade20k_food_to_foodother.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'food_other', 	'fruit_other',	'ade20k_food/ADE20K_train_ade20k_food_to_fruitother.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'food_other', 	'pizza',		'ade20k_food/ADE20K_train_ade20k_food_to_pizza.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'food_other', 	'sandwich',		'ade20k_food/ADE20K_train_ade20k_food_to_sandwich.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'food_other', 	'unlabeled',	'ade20k_food/ADE20K_train_ade20k_food_to_unlabel.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'food_other', 	'vegetation',	'ade20k_food/ADE20K_train_ade20k_food_to_vegetation.txt'),

	DatasetClassUpdateRecord('ade20k-v1', 'val',	'food_other',	'broccoli', 	'ade20k_food/ADE20K_val_ade20k_food_to_broccoli.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val',	'food_other',	'food_other', 	'ade20k_food/ADE20K_val_ade20k_food_to_food-other.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val',	'food_other',	'fruit_other', 	'ade20k_food/ADE20K_val_ade20k_food_to_fruit-other.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val',	'food_other',	'pizza', 		'ade20k_food/ADE20K_val_ade20k_food_to_pizza.txt'),

	# ADE20K_ade20k_glass (ADE20K GLASS -> UNIVERSAL WINE-GLASS)
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'wine_glass', 'bottle',	 		'ade20k_glass/ADE20K_train_ade20k_glass_to_bottle.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'wine_glass', 'cup',	 		'ade20k_glass/ADE20K_train_ade20k_glass_to_cup.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'wine_glass', 'mirror',	 		'ade20k_glass/ADE20K_train_ade20k_glass_to_mirror.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'wine_glass', 'unlabeled',	 	'ade20k_glass/ADE20K_train_ade20k_glass_to_None_of_these.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'wine_glass', 'window',	 		'ade20k_glass/ADE20K_train_ade20k_glass_to_window.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'wine_glass', 'wine_glass',	 	'ade20k_glass/ADE20K_train_ade20k_glass_to_wine_glass.txt'),

	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'wine_glass', 'cup',	 		'ade20k_glass/ADE20K_val_ade20k_glass_to_cup.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'wine_glass', 'unlabeled',	 	'ade20k_glass/ADE20K_val_ade20k_glass_to_unlabel.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'wine_glass', 'window',	 		'ade20k_glass/ADE20K_val_ade20k_glass_to_window.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'wine_glass', 'wine_glass',	 	'ade20k_glass/ADE20K_val_ade20k_glass_to_wine_glass.txt'),

	# ADE20K_ade20k_motorcyclist_bicyclist (ADE20K PERSON -> UNIVERSAL PERSON-NON-RIDER)
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'person', 'bicyclist', 	'ade20k_motorcyclist_bicyclist/ADE20K_train_ade20k_motorcyclist_bicyclist_to_bicyclist.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'person', 'motorcyclist', 'ade20k_motorcyclist_bicyclist/ADE20K_train_ade20k_motorcyclist_bicyclist_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'person', 'rider_other', 	'ade20k_motorcyclist_bicyclist/ADE20K_train_ade20k_motorcyclist_bicyclist_to_rider-other.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val',	'person',	'bicyclist', 	'ade20k_motorcyclist_bicyclist/ADE20K_val_ade20k_motorcyclist_bicyclist_to_bicyclist.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val',	'person',	'motorcyclist', 'ade20k_motorcyclist_bicyclist/ADE20K_val_ade20k_motorcyclist_bicyclist_to_motorcyclist.txt'),

	# ade20k_ade20k_nonrider (ADE20K PERSON -> UNIVERSAL PERSON-NON-RIDER)
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'person', 'bicyclist', 		'ade20k_nonrider/ade20k_train_ade20k_nonrider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'person', 'person', 'ade20k_nonrider/ade20k_train_ade20k_nonrider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val',	'person',	'person', 'ade20k_nonrider/ade20k_val_ade20k_nonrider_to_person-non-rider.txt'),

	#ade20k_ade20k_table (ADE20K TABLE -> UNIVERSAL TABLE)
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'bathroom_counter', 'ade20k_table/ade20k_train_ade20k_table_to_bathroom-counter.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'cabinet',			'ade20k_table/ade20k_train_ade20k_table_to_cabinet.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'counter_other', 	'ade20k_table/ade20k_train_ade20k_table_to_counter-other.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'desk', 			'ade20k_table/ade20k_train_ade20k_table_to_desk-and-table.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'desk', 			'ade20k_table/ade20k_train_ade20k_table_to_desk.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'kitchen_island', 	'ade20k_table/ade20k_train_ade20k_table_to_kitchen-island.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'nightstand', 		'ade20k_table/ade20k_train_ade20k_table_to_nightstand.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'unlabeled', 		'ade20k_table/ade20k_train_ade20k_table_to_None_of_these.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'table', 			'ade20k_table/ade20k_train_ade20k_table_to_table.txt'),

	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'table', 	'counter_other', 	'ade20k_table/ade20k_val_ade20k_table_to_counter-other.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'table', 	'desk', 			'ade20k_table/ade20k_val_ade20k_table_to_desk-and-table.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'table', 	'desk', 			'ade20k_table/ade20k_val_ade20k_table_to_desk.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'table', 	'kitchen_island', 	'ade20k_table/ade20k_val_ade20k_table_to_kitchen-island.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'table', 	'nightstand', 		'ade20k_table/ade20k_val_ade20k_table_to_nightstand.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'table', 	'unlabeled', 		'ade20k_table/ade20k_val_ade20k_table_to_None_of_these.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'table', 	'stool', 			'ade20k_table/ade20k_val_ade20k_table_to_stool.txt'),
	DatasetClassUpdateRecord('ade20k-v1', 'val', 	'table', 	'table', 			'ade20k_table/ade20k_val_ade20k_table_to_table.txt'),


	# Phase 3, ade20k_plaything
							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('ade20k', 'train', 'plaything_other', 'plaything_other','ade20k_plaything/ade20k_train_ade20k_plaything_to_plaything-other.txt'),
	DatasetClassUpdateRecord('ade20k', 'train', 'plaything_other', 'teddy_bear', 	 'ade20k_plaything/ade20k_train_ade20k_plaything_to_teddy-bear.txt'),

	DatasetClassUpdateRecord('ade20k', 'val', 'plaything_other', 'plaything_other',  'ade20k_plaything/ade20k_val_ade20k_plaything_to_plaything-other.txt'),
	DatasetClassUpdateRecord('ade20k', 'val', 'plaything_other', 'teddy_bear', 		 'ade20k_plaything/ade20k_val_ade20k_plaything_to_teddy-bear.txt')
]


cityscapes_update_records = [
	# cityscapes_rider, Phase 2 (CITYSCAPES RIDER -> UNVERISAL RIDER-OTHER)
							# dataset    # split  # orig u name  # new u name    	# path to txt file
	DatasetClassUpdateRecord('cityscapes', 'train', 'rider_other', 	'bicycle', 			'cityscapes_rider/cityscapes_train_cityscapes_rider_to_bicycle.txt'),
	DatasetClassUpdateRecord('cityscapes', 'train', 'rider_other', 	'bicyclist', 		'cityscapes_rider/cityscapes_train_cityscapes_rider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('cityscapes', 'train', 'rider_other', 	'motorcyclist', 	'cityscapes_rider/cityscapes_train_cityscapes_rider_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('cityscapes', 'train', 'rider_other', 	'unlabeled', 		'cityscapes_rider/cityscapes_train_cityscapes_rider_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cityscapes', 'train', 'rider_other', 	'person', 'cityscapes_rider/cityscapes_train_cityscapes_rider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('cityscapes', 'train', 'rider_other', 	'rider_other', 		'cityscapes_rider/cityscapes_train_cityscapes_rider_to_rider-other.txt'),

	DatasetClassUpdateRecord('cityscapes', 'val', 	'rider_other', 	'bicycle', 			'cityscapes_rider/cityscapes_val_cityscapes_rider_to_bicycle.txt'),
	DatasetClassUpdateRecord('cityscapes', 'val', 	'rider_other', 	'bicyclist', 		'cityscapes_rider/cityscapes_val_cityscapes_rider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('cityscapes', 'val', 	'rider_other', 	'motorcyclist', 	'cityscapes_rider/cityscapes_val_cityscapes_rider_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('cityscapes', 'val', 	'rider_other', 	'unlabeled', 		'cityscapes_rider/cityscapes_val_cityscapes_rider_to_None_of_these.txt'),
	DatasetClassUpdateRecord('cityscapes', 'val', 	'rider_other', 	'person', 'cityscapes_rider/cityscapes_val_cityscapes_rider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('cityscapes', 'val', 	'rider_other', 	'rider_other', 		'cityscapes_rider/cityscapes_val_cityscapes_rider_to_rider-other.txt')
]


bdd_update_records = [
	# bdd person, Phase 2 (BDD PERSON -> UNIVERSAL PERSON-NONRIDER)
							# dataset  # split   # orig u name 			 # new u name    	# path to txt file
	DatasetClassUpdateRecord('bdd', 	'train', 'person', 	'bicyclist', 		'bdd_person/bdd_train_bdd_person_to_bicyclist.txt'),
	DatasetClassUpdateRecord('bdd', 	'train', 'person', 	'motorcyclist', 	'bdd_person/bdd_train_bdd_person_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('bdd', 	'train', 'person', 	'unlabeled', 		'bdd_person/bdd_train_bdd_person_to_None_of_these.txt'),
	DatasetClassUpdateRecord('bdd', 	'train', 'person', 	'person', 'bdd_person/bdd_train_bdd_person_to_person-non-rider.txt'),

	DatasetClassUpdateRecord('bdd', 	'val', 	'person', 	'bicyclist', 		'bdd_person/bdd_val_bdd_person_to_bicyclist.txt'),
	DatasetClassUpdateRecord('bdd', 	'val', 	'person', 	'person', 'bdd_person/bdd_val_bdd_person_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('bdd', 	'val', 	'person', 	'unlabeled', 		'bdd_person/bdd_val_bdd_person_to_unlabel.txt'),

	# bdd_rider, Phase 2 (BDD RIDER -> UNIVERSAL RIDER-OTHER))
							# dataset  # split  # orig u name  		# new u name    	# path to txt file
	DatasetClassUpdateRecord('bdd', 	'train', 'rider_other', 	'bicyclist', 		'bdd_rider/bdd_train_bdd_rider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('bdd', 	'train', 'rider_other', 	'motorcyclist', 	'bdd_rider/bdd_train_bdd_rider_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('bdd', 	'train', 'rider_other', 	'unlabeled', 		'bdd_rider/bdd_train_bdd_rider_to_None_of_these.txt'),
	DatasetClassUpdateRecord('bdd', 	'train', 'rider_other', 	'person', 'bdd_rider/bdd_train_bdd_rider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('bdd', 	'train', 'rider_other', 	'rider_other', 		'bdd_rider/bdd_train_bdd_rider_to_rider-other.txt'),

	DatasetClassUpdateRecord('bdd', 	'val', 	'rider_other', 		'bicyclist', 		'bdd_rider/bdd_val_bdd_rider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('bdd', 	'val', 	'rider_other', 		'motorcyclist', 	'bdd_rider/bdd_val_bdd_rider_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('bdd', 	'val', 	'rider_other', 		'person', 'bdd_rider/bdd_val_bdd_rider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('bdd', 	'val', 	'rider_other',		'rider_other', 		'bdd_rider/bdd_val_bdd_rider_to_rider-other.txt')
]


idd_update_records = [
	# idd_idd_rider, Phase 2 (IDD-NEW RIDER -> UNIVERSAL RIDER-OTHER)
							# dataset  # split  	# orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'backpack', 		'idd_rider/idd_train_idd_rider_to_backpack.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'bag', 				'idd_rider/idd_train_idd_rider_to_bag.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'bicycle', 			'idd_rider/idd_train_idd_rider_to_bicycle.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'bicyclist', 		'idd_rider/idd_train_idd_rider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'box', 				'idd_rider/idd_train_idd_rider_to_box.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'motorcycle', 		'idd_rider/idd_train_idd_rider_to_motorcycle.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'motorcyclist', 	'idd_rider/idd_train_idd_rider_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'unlabeled', 		'idd_rider/idd_train_idd_rider_to_None_of_these.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'person', 'idd_rider/idd_train_idd_rider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'pole', 			'idd_rider/idd_train_idd_rider_to_pole.txt'),
	DatasetClassUpdateRecord('idd', 	'train', 	'rider_other', 	'rider_other', 		'idd_rider/idd_train_idd_rider_to_rider-other.txt'),

	DatasetClassUpdateRecord('idd', 	'val', 		'rider_other', 	'bicyclist', 		'idd_rider/idd_val_idd_rider_to_bicyclist.txt'),
	DatasetClassUpdateRecord('idd', 	'val', 		'rider_other', 	'motorcyclist', 	'idd_rider/idd_val_idd_rider_to_motorcyclist.txt'),
	DatasetClassUpdateRecord('idd', 	'val', 		'rider_other', 	'unlabeled', 		'idd_rider/idd_val_idd_rider_to_None_of_these.txt'),
	DatasetClassUpdateRecord('idd', 	'val', 		'rider_other', 	'person', 'idd_rider/idd_val_idd_rider_to_person-non-rider.txt'),
	DatasetClassUpdateRecord('idd', 	'val', 		'rider_other', 	'rider_other', 		'idd_rider/idd_val_idd_rider_to_rider-other.txt')
]


# Note: we refer to the `test` split as `val`, since `val` not easily defined/found.
sunrgbd_update_records = [
	# sunrgbd_chair
							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('sunrgbd', 'val', 'chair_other', 	'armchair', 	'sunrgbd_chair/sunrgbd_test_sunrgbd_chair_to_armchair.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'chair_other', 	'bench', 		'sunrgbd_chair/sunrgbd_test_sunrgbd_chair_to_bench.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'chair_other', 	'chair_other', 	'sunrgbd_chair/sunrgbd_test_sunrgbd_chair_to_chairother.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'chair_other', 	'seat', 		'sunrgbd_chair/sunrgbd_test_sunrgbd_chair_to_seat.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'chair_other', 	'couch', 		'sunrgbd_chair/sunrgbd_test_sunrgbd_chair_to_sofa.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'chair_other', 	'stool', 		'sunrgbd_chair/sunrgbd_test_sunrgbd_chair_to_stool.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'chair_other', 	'swivel_chair',	'sunrgbd_chair/sunrgbd_test_sunrgbd_chair_to_swivelchair.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'chair_other', 	'unlabeled', 	'sunrgbd_chair/sunrgbd_test_sunrgbd_chair_to_unlabel.txt'),
	
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'armchair', 	'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_armchair.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'bench', 		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_bench.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'cabinet',		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_cabinet.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'chair_other',  'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_chair-other.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'door', 		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_door.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'unlabeled', 	'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_None_of_these.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'ottoman', 		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_ottoman.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'seat', 		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_seat.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'couch', 		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_sofa.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'stool', 		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_stool.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'swivel_chair',	'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_swivel-chair.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'table', 		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_table.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','chair_other', 	'wall', 		'sunrgbd_chair/sunrgbd_train_sunrgbd_chair_to_wall.txt'),

	# sunrgbd_counter
	DatasetClassUpdateRecord('sunrgbd', 'val', 'counter_other', 'bathroom_counter','sunrgbd_counter/sunrgbd_test_sunrgbd_counter_to_bathroomcounter.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'counter_other', 'counter_other',	'sunrgbd_counter/sunrgbd_test_sunrgbd_counter_to_counter-other.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 'counter_other', 'unlabeled',		'sunrgbd_counter/sunrgbd_test_sunrgbd_counter_to_unlabel.txt'),

	DatasetClassUpdateRecord('sunrgbd', 'train','counter_other', 'bathroom_counter','sunrgbd_counter/sunrgbd_train_sunrgbd_counter_to_bathroom-counter.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','counter_other', 'cabinet',			'sunrgbd_counter/sunrgbd_train_sunrgbd_counter_to_cabinetmerged.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','counter_other', 'counter_other',	'sunrgbd_counter/sunrgbd_train_sunrgbd_counter_to_counter-other.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','counter_other', 'desk',		 	'sunrgbd_counter/sunrgbd_train_sunrgbd_counter_to_desk.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','counter_other', 'kitchen_island',	'sunrgbd_counter/sunrgbd_train_sunrgbd_counter_to_kitchen-island.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','counter_other', 'unlabeled',		'sunrgbd_counter/sunrgbd_train_sunrgbd_counter_to_None_of_these.txt'),

	# sunrgbd_lamp (TODO: map it to "lamp" inside of the universal tax.)
	DatasetClassUpdateRecord('sunrgbd', 'val', 	'light_other', 'lamp',			'sunrgbd_lamp/sunrgbd_test_sunrgbd_lamp_to_lamp.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 	'light_other', 'sconce',		'sunrgbd_lamp/sunrgbd_test_sunrgbd_lamp_to_sconce.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'val', 	'light_other', 'unlabeled',		'sunrgbd_lamp/sunrgbd_test_sunrgbd_lamp_to_unlabel.txt'),

	DatasetClassUpdateRecord('sunrgbd', 'train','light_other', 'chandelier',	'sunrgbd_lamp/sunrgbd_train_sunrgbd_lamp_to_chandelier.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','light_other', 'lamp',			'sunrgbd_lamp/sunrgbd_train_sunrgbd_lamp_to_lamp.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','light_other', 'light_other',	'sunrgbd_lamp/sunrgbd_train_sunrgbd_lamp_to_light-other.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','light_other', 'unlabeled',		'sunrgbd_lamp/sunrgbd_train_sunrgbd_lamp_to_None_of_these.txt'),
	DatasetClassUpdateRecord('sunrgbd', 'train','light_other', 'sconce',		'sunrgbd_lamp/sunrgbd_train_sunrgbd_lamp_to_sconce.txt')
]




mapillary_update_records = [
	# mapillarypublic_water
							# dataset  # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'water_other', 'fountain', 	 'mapillarypublic_water/mapillarypublic_train_water_to_fountain.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'water_other', 'unlabeled',  'mapillarypublic_water/mapillarypublic_train_water_to_None_of_these.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'water_other', 'river_lake', 'mapillarypublic_water/mapillarypublic_train_water_to_river-lake.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'water_other', 'sea', 	 	 'mapillarypublic_water/mapillarypublic_train_water_to_sea.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'water_other', 'water_other','mapillarypublic_water/mapillarypublic_train_water_to_water-other.txt'),
	
	DatasetClassUpdateRecord('mapillary-public65', 'val', 'water_other', 'fountain',	 'mapillarypublic_water/mapillarypublic_val_water_to_fountain.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'val', 'water_other', 'river_lake',	 'mapillarypublic_water/mapillarypublic_val_water_to_river-lake.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'val', 'water_other', 'sea',			 'mapillarypublic_water/mapillarypublic_val_water_to_sea.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'val', 'water_other', 'water_other',	 'mapillarypublic_water/mapillarypublic_val_water_to_water-other.txt'),

	# mapillarypublic_groundanimal
									# dataset      # split  # orig u name  	# new u name    	# path to txt file
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'dog', 'animal_other',	'mapillarypublic_groundanimal/mapillaryacademic_train_groundanimal_to_animalother.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'dog', 'bird', 		'mapillarypublic_groundanimal/mapillaryacademic_train_groundanimal_to_bird.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'dog', 'cat', 			'mapillarypublic_groundanimal/mapillaryacademic_train_groundanimal_to_cat.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'dog', 'cow', 			'mapillarypublic_groundanimal/mapillaryacademic_train_groundanimal_to_cow.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'dog', 'dog', 			'mapillarypublic_groundanimal/mapillaryacademic_train_groundanimal_to_dog.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'dog', 'horse', 		'mapillarypublic_groundanimal/mapillaryacademic_train_groundanimal_to_horse.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'train', 'dog', 'unlabeled',	'mapillarypublic_groundanimal/mapillaryacademic_train_groundanimal_to_none.txt'),

	DatasetClassUpdateRecord('mapillary-public65', 'val', 'dog', 'bird', 		'mapillarypublic_groundanimal/mapillaryacademic_val_groundanimal_to_bird.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'val', 'dog', 'dog', 		'mapillarypublic_groundanimal/mapillaryacademic_val_groundanimal_to_dog.txt'),
	DatasetClassUpdateRecord('mapillary-public65', 'val', 'dog', 'unlabeled', 	'mapillarypublic_groundanimal/mapillaryacademic_val_groundanimal_to_none.txt')
]





