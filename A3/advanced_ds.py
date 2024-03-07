## WORKING ON IT ##

def merge_into_sentences(sent_list):

    word_sentlist = []
    for sent in sent_list:
        featdict = {}
        token_list=[]
        label_list=[]
        pred_list=[]
        for d in sent:
            if d["V"] != '_':
                token_list.append("Mark")
                label_list.append(label_dict["_"])
                token_list.append(d["form"])
                label_list.append(label_dict[d["ARG"]])
                token_list.append("Mark")
                label_list.append(label_dict["_"])
                pred_list.append(d["form"])
            else:
                token_list.append(d["form"])
                label_list.append(label_dict[d["ARG"]])

        featdict['tokens'],featdict['srl_arg_tags'],featdict['pred'] = token_list,label_list,pred_list
        word_sentlist.append(featdict)
    return word_sentlist