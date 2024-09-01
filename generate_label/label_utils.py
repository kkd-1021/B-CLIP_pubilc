def get_person_name(name):
    if (name[-1] =='0' or name[-1] =='1' or name[-1] =='2'
            or name[-1] =='3' or name[-1] =='4'or name[-1] =='5' or name[-1] =='6'):
        name=name[:-1]

    if name[-1].isdigit() is True:
        print("file name error")
        exit(1)
    return name


def find_value_from_xlsx(feat_name:str,excel_data,name:str):
    for idx in range(excel_data.shape[0]):
        if excel_data['name'][idx]==name:
            return excel_data[feat_name][idx]

    print('not found')
    #exit(1)