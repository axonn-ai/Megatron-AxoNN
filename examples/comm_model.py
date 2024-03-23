import pandas as pd

def get_bw(ip, my, machine="perlmutter"):
    if machine == "perlmutter":
        if ip==1:
            if my==2:
                return 76
            elif my==4:
                return 225
            elif my>=8:
                return 80
        elif ip == 2:
            if my == 2:
                return 76
            elif my >= 4:
                return 40
        elif ip >= 4:
            return 20
    elif machine == "frontier":
        if ip==1:
            if my==2:
                return 129
            elif my==4:
                return 52
            elif my==8:
                return 135
            else:
                return 80 / 2 # 34.031
        elif ip == 2:
            if my == 2:
                return 50
            elif my == 4:
                return 72
            else:
                return 40 / 2  # 20.62
        elif ip == 4:
            if my == 2:
                return 36 
            else:
                return 20 / 2 # 10.18
        elif ip >=8: 
            return 10 / 2 # 


def model_v2(B, S, K, H, Gd, Gr, Gc, Gdata, machine="perlmutter", transpose=False, grad_acc=1):
    dp_comm = 4 * (Gdata-1) / Gdata * K * H * H / (Gc*Gr*Gd) * 2 /1024/1024/1024
    depth_tensor_comm = 3/2 * 2 * (Gd-1) / Gd * K * H * H / (Gc*Gr) * 2 /1024/1024/1024
    depth_tensor_comm *= grad_acc

    if not transpose:
        row_tensor_comm = 2 * (Gr-1) / Gr * (B/Gdata/Gd * S * H/Gc ) * 2 /1024/1024/1024
        col_tensor_comm = 2 * 2 * (Gc-1) / Gc * (B/Gdata/Gd * S * K * H/Gr ) * 2 /1024/1024/1024
    else:
        row_tensor_comm = 2 * 2 * (Gr-1) / Gr * (B/Gdata/Gd * S * K * H/Gc) * 2 /1024/1024/1024
        col_tensor_comm = 2 * (Gc-1) / Gc * (B/Gdata/Gd * S * H/Gr ) * 2 /1024/1024/1024

    col_time = row_time = depth_time = data_time = 0 
    ip=1
    if Gc > 1:
        col_bw = get_bw(ip, Gc, machine)
        if col_bw is None:
            return None
        col_time = col_tensor_comm / col_bw 
    ip*= Gc 
    if Gr > 1:
        row_bw = get_bw(ip, Gr, machine)
        if row_bw is None:
            return None
        row_time = row_tensor_comm / row_bw 

    ip *= Gr 
    
    if Gd > 1:
        depth_bw = get_bw(ip, Gd, machine)
        if depth_bw is None:
            return None
        depth_time = depth_tensor_comm / depth_bw 

    ip *= Gd

    if Gdata > 1: 
        data_bw = get_bw(ip, Gdata, machine)
        if data_bw is None:
            return None
        data_time = dp_comm / data_bw 


    return col_time + row_time + depth_time + data_time



def get_configs_for_transformer(
        global_batch_size_in_samples,
        sequence_length,
        num_layers,
        hidden_size, 
        GPUs,
        minimum_degree_of_tensor_parallelism,
        model_version="v2",
        topk=5,
        no_dp=False,
        machine="perlmutter",
        grad_acc=1
):
    S=sequence_length
    K=3
    B=global_batch_size_in_samples
    H=hidden_size
    G=GPUs
    min_tp=minimum_degree_of_tensor_parallelism


    range = []
    i=0
    while 2**i <=G:
        range.append(2**i)
        i+=1

    data = {}
    for Gc in range:
        for Gr in range:
            for Gd in range:
                for Gdata in range:
                    if Gc*Gr*Gd*Gdata == G and Gc*Gr*Gd>=min_tp and B%(Gdata*Gd)==0 and (not no_dp or Gdata==1):
                        #data[(Gc,Gr,Gd,Gdata)] =
                        a = model_v2(B, S, 3, H, Gd, Gr, Gc, Gdata, machine, grad_acc=grad_acc) 
                        b = model_v2(B, S, 1, H, Gd, Gr, Gc, Gdata, machine, transpose=True, grad_acc=grad_acc) 
                        c = model_v2(B, S, 4, H, Gd, Gr, Gc, Gdata, machine, grad_acc=grad_acc) 
                        d = model_v2(B, S, 4, H, Gd, Gr, Gc, Gdata, machine, transpose=True, grad_acc=grad_acc)
                        if a is None or b is None or c is None or d is None:
                            continue
                        else:
                            data[(Gc,Gr,Gd,Gdata)] = a + b + c + d
                        
                        
    sorted_configs = sorted(data.items(), key=lambda x:x[1])
    
    keys = "Gr","Gc","Gd","Gdata", "Comm-Time(s)"#,"Total-Time(s)"
    data = []
    for (Gc, Gr, Gd, Gdata), comm_time in sorted_configs[:topk]:
        comm_time = comm_time * num_layers
        data.append([Gr, Gc, Gd, Gdata, comm_time]), #total_time])

    df = pd.DataFrame(data, columns=keys)
    return df
