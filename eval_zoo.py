import subprocess

if __name__ == "__main__":
    log = "Enero_3top_15_B_PATH_LINK_TEST_kp"
    subprocess.call(["python", "eval_on_zoo_topologies.py",
                     "-max_edge", "100", "-min_edge", "2",
                     "-max_nodes", "30", "-min_nodes", "1",
                     "-n", "2",
                     "-d", f"./Logs/exp{log}Logs.txt"])
    subprocess.call(["python", "figure_9.py", "-d", log, "-p", "../Enero_datasets/rwds-results-1-link_capacity-unif-05-1-zoo"])
