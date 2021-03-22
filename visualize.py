import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import os
import json


from utils.box_ops import calculate_ious
CLASSNAMES = ['i1', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i2', 'i3', 'i4', 'i5', 'il100', 'il110', 'il50', 'il60', 'il70', 'il80', 'il90', 'ip', 'p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20', 'p21', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pb', 'pc', 'pg', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.5', 'ph3.8', 'ph4', 'ph4.2', 'ph4.3', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl5', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pm10', 'pm13', 'pm15', 'pm1.5', 'pm2', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55', 'pm8', 'pn', 'pne', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'ps', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'w1', 'w10', 'w12', 'w13', 'w16', 'w18', 'w20', 'w21', 'w22', 'w24', 'w28', 'w3', 'w30', 'w31', 'w32', 'w34', 'w35', 'w37', 'w38', 'w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w5', 'w50', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w62', 'w63', 'w66', 'w8', 'i6', 'i7', 'i8', 'i9', 'p29', 'w29', 'w33', 'w36', 'w39', 'w4', 'w40', 'w51', 'w52', 'w53', 'w54', 'w6', 'w61', 'w64', 'w65', 'w67', 'w7', 'w9', 'pd', 'pe', 'pnl', 'w11', 'w14', 'w15', 'w17', 'w19', 'w2', 'w23', 'w25', 'w26', 'w27', 'pm2.5', 'ph4.4', 'ph3.3', 'ph2.6', 'i4l', 'i2r', 'im', 'wc', 'pcr', 'pcl', 'pss', 'pbp', 'p1n', 'pbm', 'pt', 'pn-2', 'pclr', 'pcs', 'pcd', 'iz', 'pmb', 'pdd', 'pctl', 'ph1.8', 'pnlc', 'pmblr', 'phclr', 'phcs', 'pmr']

def draw_box(img,box,text,color):
    box = [int(x) for x in box]
    img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=color, thickness=1)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img 


def draw_boxes(img,boxes,labels,classnames,scores=None, color=(0,0,0)):
    if scores is None:
        scores = ['']*len(labels) 
    for box,score,label in zip(boxes,scores,labels):
        box = [int(i) for i in box]
        text = classnames[label-1]+(f': {score:.2f}' if not isinstance(score,str) else score)
        img = draw_box(img,box,text,color)
    return img
def draw_result_boxes(img,boxes,labels,scores=None, color=(0,0,0)):
    if scores is None:
        scores = ['']*len(labels) 
    for box,score,label in zip(boxes,scores,labels):
        box = [int(i) for i in box]
        text = label+(f': {score:.2f}' if not isinstance(score,str) else score)
        img = draw_box(img,box,text,color)
    return img

def visualize_result(img_file,
                     pred_boxes,
                     pred_scores,
                     pred_labels,
                     gt_boxes,
                     gt_labels,
                     classnames,
                     iou_thresh=0.5,
                     miss_color=(255,0,0),
                     wrong_color=(0,255,0),
                     surplus_color=(0,0,255),
                     right_color=(0,255,255)):
    
    img = cv2.imread(img_file)

    detected = [False for _ in range(len(gt_boxes))]
    miss_boxes = []
    wrong_boxes = []
    surplus_boxes = []
    right_boxes = []

    # sort the box by scores
    ind = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[ind,:]
    pred_scores = pred_scores[ind]
    pred_labels = pred_labels[ind]

    # add background
    classnames = ['background']+classnames

    for box,score,label in zip(pred_boxes,pred_scores,pred_labels):
        ioumax = 0.
        if len(gt_boxes)>0:
            ioumax,jmax = calculate_ious(gt_boxes,box)
        if ioumax>iou_thresh:
            if not detected[jmax]:
                detected[jmax]=True
                if label == gt_labels[jmax]:
                    right_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
                else:
                    wrong_boxes.append((box,f'{classnames[label]}->{classnames[gt_labels[jmax]]}'))
            else:
                surplus_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
        else:
            surplus_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
    
    for box,label,d in zip(gt_boxes,gt_labels,detected):
        if not d:
            miss_boxes.append((box,f'{classnames[label]}'))
    
    colors = [miss_color]*len(miss_boxes) + [wrong_color]*len(wrong_boxes) + [right_color]*len(right_boxes) + [surplus_color]*len(surplus_boxes)

    boxes = miss_boxes + wrong_boxes + right_boxes + surplus_boxes
    
    for (box,text),color in zip(boxes,colors):
        img = draw_box(img,box,text,color)
    
    # draw colors
    colors = [right_color,wrong_color,miss_color,surplus_color]
    texts = ['Detect Right','Detect Wrong Class','Missed Ground Truth','Surplus Detection']
    for i,(color,text) in enumerate(zip(colors,texts)):
        img = cv2.rectangle(img=img, pt1=(0,i*30), pt2=(60,(i+1)*30), color=color, thickness=-1)
        img = cv2.putText(img=img, text=text, org=(70,(i+1)*30-5), fontFace=0, fontScale=0.8, color=color, thickness=2)
    return img


def find_dir(data_dir,img_id):
    t_f = f"{data_dir}/test/{img_id}.jpg"
    tt_f = f"{data_dir}/train/{img_id}.jpg"
    o_f = f"{data_dir}/other/{img_id}.jpg"
    r_f = f"{data_dir}/TEST_A/{img_id}.jpg"
    if os.path.exists(tt_f):
        return tt_f
    elif os.path.exists(t_f):
        return t_f
    elif os.path.exists(o_f):
        return o_f
    assert False,f"{img_id}.jpg is not exists"

def save_visualize_image(data_dir,img_id,pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,classnames):
    img_file =  find_dir(data_dir,img_id)

    img = visualize_result(img_file,pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,classnames)

    os.makedirs('test_imgs',exist_ok=True)
    cv2.imwrite(f'test_imgs/{img_id}.jpg',img)

def save_result_image(data_dir, img_id, pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,classnames):
    img_file = find_dir(data_dir, img_id)

    img = visualize_result(img_file, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, classnames)
    os.makedirs('result_imgs', exist_ok=True)
    cv2.imwrite(f'result_imgs/{img_id}.jpg', img)



def main():
    epoch = 20
    data_dir = "/Myhome/datasets/tt100k_2021/TEST_A"
    result_dir = f"/Myhome/datasets/tt100k_2021/trafficsign_{epoch}/test_result.json"
    save_path = f"/Myhome/datasets/tt100k_2021/trafficsign_{epoch}/pic/"
    r_annos = json.load(open(result_dir))
    for img in r_annos.items():
        path = data_dir + "/" + img[0]
        objs = img[1]
        img_file = cv2.imread(path)
        boxes = [[o['bbox']['xmin'],o['bbox']['ymin'],o['bbox']['xmax'],o['bbox']['ymax']] for o in objs]
        labels = [o['category'] for o in objs]
        img_file = draw_result_boxes(img_file, boxes, labels)
        cv2.imwrite(f'{save_path}/{img[0]}', img_file)
        break
if __name__ == "__main__":
    main()
