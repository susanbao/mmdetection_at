# python active_testing.py --model_data_type DETR_COCO --data_type image --ck_nums 10
# python active_testing.py --model_data_type DETR_COCO --data_type region --ck_nums 10
# python active_testing.py --model_data_type DFDETR_COCO --data_type image --ck_nums 10
# python active_testing.py --model_data_type DFDETR_COCO --data_type region --ck_nums 10
# python active_testing.py --model_data_type DINO_COCO --data_type image --ck_nums 7
# python active_testing.py --model_data_type DINO_COCO --data_type region --ck_nums 7

# python active_testing.py --model_data_type DFDETR_COCO_32 --data_type image --ck_nums 7
# python active_testing.py --model_data_type DFDETR_COCO_32 --data_type region --ck_nums 7

python active_testing.py --model_data_type SSD_COCO --data_type image --ck_nums 10
python active_testing.py --model_data_type SSD_COCO --data_type region --ck_nums 10

python active_testing_LLAL.py --model_data_type SSD_COCO --data_type image --loss_range image