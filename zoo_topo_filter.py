import os

dataset_folder = "../Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo"
enero_zoo_topo_set = "../Enero_datasets/rwds-results-1-link_capacity-unif-05-1-zoo/SP_3top_15_B_NEW"
enero_zoo_topo = os.listdir(enero_zoo_topo_set)

for topo in os.listdir(dataset_folder):
    if topo not in enero_zoo_topo:
        os.system("rm -rf %s" % (dataset_folder + '/' + topo))
        print(dataset_folder + '/' + topo)
