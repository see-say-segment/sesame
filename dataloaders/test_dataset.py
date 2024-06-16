import glob
import os
import random

import cv2
import numpy as np
import torch
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .reason_seg_dataset import ReasonSegDataset
from .qa_template import SHORT_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE, NEG_ANSWER_TEMPLATE, CORRECT_ANSWER_TEMPLATE, LONG_QUESTION_TEMPLATE, LONG_ANSWER_TEMPLATE
from .trainval_dataset import collate_fn_val


collate_fn_test = collate_fn_val


class TestReferDataset(ReferSegDataset):
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        train_test_split="val",
        datasetname="fprefcoco",
    ):
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE
        self.neg_answer_list = NEG_ANSWER_TEMPLATE
        self.correct_answer_list = CORRECT_ANSWER_TEMPLATE
        # Load dataset
        self.ds = ds = datasetname

        data_dir = os.path.join(self.base_image_dir, "refer_seg")
        split_by = self.determine_split_by(ds)
        refer_api = REFER(data_dir, ds, split_by)
        ref_ids_test = refer_api.getRefIds(split=train_test_split)
        images_ids_test = refer_api.getImgIds(ref_ids=ref_ids_test)
        refs_test = refer_api.loadRefs(ref_ids=ref_ids_test)
        self.test_dataset = self.prepare_dataset(ds, refer_api, images_ids_test, refs_test, data_dir)
        print("data length = ", len(self.test_dataset["images"]))

    def __len__(self):
        return len(self.test_dataset["images"])

    def select_dataset_and_image(self, idx):
        """Selects a random dataset and an image from it."""
        refer_seg_ds = self.test_dataset
        images, annotations, img2refs = refer_seg_ds["images"], refer_seg_ds["annotations"], refer_seg_ds["img2refs"]
        
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        return self.ds, image_info, refs, annotations

    def process_referring_expressions(self, refs):
        # Load referring expression info.
        Q_sents = []
        gt_sents = []
        ann_ids = []
        ref_ids = []
        sent_ids = []
        exists = []
        for ref in refs:
            for idx, sent in enumerate(ref["sentences"]):
                text = sent["sent"]
                Q_sents.append(text)
                gt_sents.append(sent.get("gt_sent", ""))
                ann_ids.append(ref["ann_id"])
                ref_ids.append(ref["ref_id"])
                sent_ids.append(idx)
                if "is_false_premise" in sent:
                    exists.append(not sent["is_false_premise"])
                elif "exist" in sent:
                    exists.append(sent["exist"])
                else:
                    exists.append(True)
        return Q_sents, gt_sents, ann_ids, exists, ref_ids, sent_ids

    def __getitem__(self, idx):
        # get one sample
        ds, image_info, refs, annotations = self.select_dataset_and_image(idx)
        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_info["file_name"])
        # load referring expression
        Q_sents, A_sents, ann_ids, exists, ref_ids, sent_ids = self.process_referring_expressions(refs)
        # create conversation Q/A (convert it to LLaVA type)
        conversations = self.create_conversations(ds, Q_sents, A_sents, exists, load_answer=False)
        # load segmentation masks
        masks = self.load_segmentation_masks(image_info, annotations, sam_input_shape, ann_ids, exists, include_nonexist=True)
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        # print(masks.shape[1] == sam_mask_shape[2] and masks.shape[2] == sam_mask_shape[3], flush=True)
        return (
            image_info["file_name"],    # filename
            image,                      # raw image (for SAM)
            image_clip,                 # image clip feature (for LMMs)
            conversations,              # QA
            masks,                      # segmentation GT
            sam_mask_shape,             # input / output shape for SAM
            exists,                     # object existence
            ref_ids,                    # ref id (useless now)
            sent_ids                    # sent id (useless now)
        )



class TestReasoningDataset(ReasonSegDataset):
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        train_test_split="val",
        datasetname="ReasonSeg",
    ):
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.long_question_list = LONG_QUESTION_TEMPLATE
        self.answer_list = LONG_ANSWER_TEMPLATE
        # load dataset
        reason_seg_data, splits = datasetname, train_test_split
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

    def __len__(self):
        return len(self.reason_seg_data[0])

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        image_path = images[idx]
        json_path = jsons[idx]

        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
        # Get sents and segmentation maps
        img = cv2.imread(image_path)[:, :, ::-1]
        mask, sents, fp_qa, is_sentence = get_mask_from_json(json_path, img)
        # Sampling
        # assert len(sents) == len(fp_qa) == 1
        masks = [
            (mask == 1).astype(np.float32), np.zeros_like(mask).astype(np.float32)
        ]

        # Create Q/A Data
        conversations = []
        # True premise question
        conv = conversation_lib.default_conversation.copy()
        if is_sentence:
            question_template = random.choice(self.long_question_list)
            Q_sent = question_template.format(sent=sents[0])
        else:
            question_template = random.choice(self.short_question_list)
            Q_sent = question_template.format(class_name=sents[0].lower())
        conv.append_message(conv.roles[0], Q_sent)
        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())

        # False premise question
        conv = conversation_lib.default_conversation.copy()
        # True premise question
        if fp_qa[0][1] is True:
            question_template = random.choice(self.long_question_list)
            neg_Q_sent = question_template.format(sent=fp_qa[0][0])
        else:
            question_template = random.choice(self.short_question_list)
            neg_Q_sent = question_template.format(class_name=fp_qa[0][0])
        conv.append_message(conv.roles[0], neg_Q_sent)
        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        # Exists and segmentation masks
        exists = [True, False]
        masks = torch.from_numpy(np.stack(masks, axis=0))
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        ref_ids = [int(idx), int(idx)]
        sent_ids = [0, 1]
        return (
            image_path,         # filename
            image,              # raw image (for SAM)
            image_clip,         # image clip feature (for LMMs)
            conversations,      # QA
            masks,              # segmentation GT
            sam_mask_shape,     # input / output shape for SAM
            exists,             # object existence
            ref_ids,            # ref id (useless now)
            sent_ids            # sent id (useless now)
        )