import subprocess

if __name__ == "__main__":
    topo = ["NEW_EliBackbone/EVALUATE", "NEW_Janetbackbone/EVALUATE", "NEW_HurricaneElectric/EVALUATE"]
    log = "Enero_3top_15_B_PATH_LINK_TEST_kp"
    subprocess.call(["python", "parse_PPO.py", "-d", "./Logs/exp" + log + "Logs.txt"])
    for t in topo:
        subprocess.call(["python", "eval_on_single_topology.py",
                         "-max_edge", "100", "-min_edge", "5",
                         "-max_nodes", "30", "-min_nodes", "1",
                         "-n", "2",
                         "-f1", "results_single_top", "-f2", t,
                         "-d", f"./Logs/exp{log}Logs.txt"])
    subprocess.call(["python", "figures_5_and_6.py", "-d", log])
