import subprocess

if __name__ == "__main__":
    topo = ["LinkFailure_EliBackbone", "LinkFailure_Janetbackbone", "LinkFailure_HurricaneElectric"]
    log = "Enero_3top_15_B_PATH_LINK_TEST_kp"
    for t in topo:
        subprocess.call(["python", "eval_on_link_failure_topologies.py",
                         "-max_edge", "100", "-min_edge", "5",
                         "-max_nodes", "30", "-min_nodes", "1",
                         "-n", "2",
                         "-d", f"./Logs/exp{log}Logs.txt",
                         "-f", t])
        subprocess.call(["python", "figure_8.py", "-d", log,
                         "-num_topologies", "20",
                         "-f", f"../Enero_datasets/dataset_sing_top/LinkFailure/rwds-{t}"])

