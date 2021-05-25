import resnet
import dla
import hardnet
import torch
import timm
import time
import cspdarknet

def benchmark(model, input_size=(608, 608), times=500):
    with torch.no_grad():
        model.eval()
        
        input = torch.rand(1,3,input_size[0],input_size[1]).cuda()
        model(input) #warmup
        model(input) #warmup
        avg_time = 0

        for i in range(0, times):
            input = torch.rand(1,3,input_size[0],input_size[1]).cuda()
            torch.cuda.synchronize()
            t1 = time.time()
            model(input)
            torch.cuda.synchronize()
            t2 = time.time()

            avg_time += (t2 - t1)

        avg_time /= times
        return avg_time

torch.backends.cudnn.benchmark = True

model_names = timm.list_models(pretrained=False)
with open("bench_x500.txt", "w") as f:
    for model_name in model_names:
        model = timm.create_model(model_name, pretrained=False).cuda()
        model.eval()
        print(model_name, " ", benchmark(model))
        f.write(model_name+" "+str(benchmark(model))+"\n")
