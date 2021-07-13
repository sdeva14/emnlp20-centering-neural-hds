import os
import pandas as pd

import xml.etree.ElementTree as ET

data_dir = "data"


import stanza  # stanford library for tokenizer
tokenizer_stanza = stanza.Pipeline('en', processors='tokenize', use_gpu=True)


def make_whole_pd(label_file, out_name):

    list_tid = []
    list_score = []
    list_title = []
    list_abst = []
    list_text = []
    list_date = []
    list_lead = []

    ind = 0
    with open(label_file) as file_in:
        # 1999_01_12_1076469.xml
        for line in file_in:
            filename = os.path.splitext(line)[0]
            splitted = filename.split("_")
            year = splitted[0]
            month = splitted[1]
            day = splitted[2]
            tid = splitted[3]

            # print(filename)

            list_tid.append(tid)


            if "good" in label_file:
                list_score.append(2)
            else:
                list_score.append(1)
                

            date = year+"_"+month+"_"+day
            list_date.append(date)

            data_path = os.path.join(data_dir, year, month, day, tid)
            data_path += ".xml"

            # print(data_path)
            ## xml parsing

            # <title>, <abstract>, <block class=lead_paragraph>, <block class=full_text>, <block class="online_lead_paragraph"> 
            tree = ET.parse(data_path)
            root = tree.getroot()

            title_str = ""
            for title_node in root.iter('title'):
                if title_node.text is not None:
                    title_str = title_node.text
            list_title.append(title_str)

            abst = ""
            for abstract in root.iter('abstract'):
                for p in abstract.iter('p'):
                    if p.text is not None:
                        abst += p.text
            list_abst.append(abst)

            contain_lead = False
            for body in root.iter('body.content'):
                for block in body.iter('block'):
                    attrib = block.attrib
                    if attrib['class'] == 'lead_paragraph':
                        contain_lead = True
                        lead_para = ""
                        for p in block.iter('p'):
                            lead_para += p.text.strip()
                            lead_para += "<split3>"
                        lead_para = lead_para[:lead_para.rfind("<split3>")]
                        list_lead.append(lead_para)

                    if attrib['class'] == 'full_text':
                        if contain_lead == False:
                            list_lead.append("")

                        full_txt = ""
                        # for p in block.iter('p'):
                        #     full_txt += p.text.strip()
                        #     full_txt += "<split3>"
                        # full_txt = full_txt[:full_txt.rfind("<split3>")]
                        # list_text.append(full_txt)

                        # stanza tokenize version

                        for p in block.iter('p'):
                            full_txt += p.text.strip()
                        
                        
                        doc_stanza = tokenizer_stanza(full_txt)
                        cur_sents = [sentence.text for sentence in doc_stanza.sentences]
                        
                        tokenize_text = ""
                        for cur_sent in cur_sents:
                            if len(cur_sent) > 1:
                                tokenize_text += cur_sent + "<split2>"

                        tokenize_text = tokenize_text[:tokenize_text.rfind("<split2>")]

                        list_text.append(tokenize_text)
            # end for body
        

            ind += 1
            if ind % 500 == 0:
                print(ind)

               

    map_data = dict()
    map_data["tid"] = list_tid
    map_data["score"] = list_score
    map_data["title"] = list_title
    map_data["abstract"] = list_abst
    map_data["lead_para"] = list_lead
    map_data["text"] = list_text

    print(len(list_tid))
    print(len(list_score))
    print(len(list_title))
    print(len(list_abst))
    print(len(list_lead))
    print(len(list_text))

    pd_data = pd.DataFrame(map_data)
    pd_data.to_csv(out_name, index=None)



if __name__ == "__main__":

    make_whole_pd("verygood_trn.list", "elisa_good_train.csv")
    make_whole_pd("verygood_dev.list", "elisa_good_valid.csv")
    make_whole_pd("verygood_tst.list", "elisa_good_test.csv")

    make_whole_pd("typical_trn.list", "elisa_typical_train.csv")
    make_whole_pd("typical_dev.list", "elisa_typical_valid.csv")
    make_whole_pd("typical_tst.list", "elisa_typical_test.csv")
