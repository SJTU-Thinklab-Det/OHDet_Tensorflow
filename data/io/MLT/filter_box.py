import os


filter_thres = 0.45
res_files = os.listdir('../../../tools/test_mlt/RetinaNet_MLT_CSL_2x_20201007')
filter_res_path = '../../../tools/test_mlt/RetinaNet_MLT_CSL_2x_20201007_{}'.format(filter_thres)

if not os.path.exists('../../../tools/test_mlt/RetinaNet_MLT_CSL_2x_20201007_{}'.format(filter_thres)):
    os.makedirs('../../../tools/test_mlt/RetinaNet_MLT_CSL_2x_20201007_{}'.format(filter_thres))

for rf in res_files:
    fr = open('../../../tools/test_mlt/RetinaNet_MLT_CSL_2x_20201007/{}'.format(rf), 'r')
    fw = open('../../../tools/test_mlt/RetinaNet_MLT_CSL_2x_20201007_{}/{}'.format(filter_thres, rf), 'w')
    lines = fr.readlines()
    for line in lines:
        if float(line.split(',')[-1].split('\n')[0])>filter_thres:
            fw.write(line)
    fr.close()
    fw.close()