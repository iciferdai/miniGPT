from myTrans.base_params import *
import pprint
import re

def process_data(txt_list):
    full_str = ''
    for chapter in txt_list:
        for s in chapter:
            full_str += s
    txt_list = list(full_str)
    cn_set = set(txt_list)

    id = CUS_START_ID
    t_dict = dict()
    ig_dict = dict()

    for i in cn_set:
        i_count = txt_list.count(i)
        if i_count >= UNK_THRESHOLD:
            t_dict[i] = id
            id += 1
        else:
            ig_dict[i] = i_count

    t_ig_dict = dict()
    for k, v in ig_dict.items():
        if v in t_ig_dict.keys():
            t_ig_dict[v].append(k)
        else:
            t_ig_dict[v] = [k]

    print(f"length: {len(txt_list)}|{len(full_str)}|{len(cn_set)}|{len(t_dict)}")
    print("CN set: ")
    print(cn_set)
    print("VOCAB Dict:")
    print(t_dict)
    print("Ignore Dict:")
    print(t_ig_dict)

def calculate_data(txt_list):
    full_str = ''
    for chapter in txt_list:
        for s in chapter:
            full_str += s
    txt_list = list(full_str)
    cn_set = set(txt_list)
    print(f"length: {len(txt_list)}|{len(full_str)}|{len(cn_set)}")

    t_dict = dict()
    for i in cn_set:
        t_dict[i] = txt_list.count(i)

    print("VOCAB Calculate:")
    print(t_dict)

    num=0
    for v in t_dict.values():
        if v>3: num+=1
    print(f"num > 3: {num}")
    num=0
    for v in t_dict.values():
        if v>5: num+=1
    print(f"num > 5: {num}")
    num=0
    for v in t_dict.values():
        if v>10: num+=1
    print(f"num > 10: {num}")
    num=0
    for v in t_dict.values():
        if v>20: num+=1
    print(f"num > 20: {num}")

def process_ori(file_name):
    novel_path = './data/' + file_name
    try:
        with open(novel_path, 'rb') as f:
            f.seek(0)
            chapters = []
            chapter = []
            for line in f:
                line_str = line.decode('utf-8').strip()
                if not line_str: continue
                if line_str[0:2] == '正文':
                    if chapter: chapters.append(chapter)
                    chapter = []
                    continue
                chapter.append(line_str)
            chapters.append(chapter)
            return chapters
    except Exception as e:
        logging.error(f"read file Error: {e}", exc_info=True)
    return None

def pre_process_data(txt_list):
    out_list = []
    # 去重
    allowed_pattern = r'[\u4e00-\u9fa5。！？；，：、…—”“（）《》0-9]'
    invalid_pattern = r'[^\u4e00-\u9fa5。！？；，：、…—”“（）《》0-9]'
    for chapter in txt_list:
        new_chapter = []
        for s in chapter:
            s = s.replace('!', '！').replace(' ', '').replace('‘', '“').replace('’', '”')
            #invalid_t = re.findall(invalid_pattern, s)
            #if invalid_t: print(f'find invalid_t {invalid_t}; from {s}')
            new_chapter.append(s)
        out_list.append(new_chapter)

    return out_list

if __name__ == '__main__':
    print('----------------Novel 4--------------------')
    txt_list = process_ori('novel_4.txt')
    #pprint.pprint(txt_list)
    after_txt = pre_process_data(txt_list)
    #process_data(after_txt)
    #pprint.pprint(after_txt)
    calculate_data(after_txt)


