import matplotlib.pyplot as plt
import numpy as np 

input_data = [[255,255,255],[0,255,0],[255,0,0]]
input_data =  np.array(input_data)


def ave_pool(input, kernel_size, stride ,padding=0):
    h,w = input.shape
    print(h,w)
    #floor((W_{in} + 2padding[1] - dilation[1](kernel_size[1] - 1) - 1)/stride[1] + 1
    output_size = np.floor((h + 2*padding - (kernel_size - 1) - 1 )/ stride) + 1
    output_size = int(output_size)
    output = np.ones([int(output_size),int(output_size)])*255
    for ph in range(output_size):
        for pw in range(output_size):
            hstart = ph *stride - padding
            wstart = pw * stride -padding
            hend = min(hstart + kernel_size, h)
            wend = min(wstart + kernel_size , w)
            #print(hstart)
            hstart = max(hstart,0)
            wstart = max(wstart, 0)
            sum = 0.0
            i = hstart
            while(i < hend):
                j = wstart
                while(j < wend):
                    sum += input[i][j]
                    #print(i,j,input[i][j])
                    j+=1
                i+=1
            ave = sum / kernel_size**2
            #print(ph,pw,hstart,hend,wstart,wend,ave)
            
            output[ph][pw] = ave
    
    return output


def max_pool(input, kernel_size, stride ,padding=0):
    h,w = input.shape
    #print(h,w)
    #floor((W_{in} + 2padding[1] - dilation[1](kernel_size[1] - 1) - 1)/stride[1] + 1
    output_size = np.floor((h + 2*padding - (kernel_size - 1) - 1 )/ stride) + 1
    output_size = int(output_size)
    output = np.ones([int(output_size),int(output_size)])*255
    for ph in range(output_size):
        for pw in range(output_size):
            hstart = ph *stride - padding
            wstart = pw * stride -padding
            hend = min(hstart + kernel_size, h)
            wend = min(wstart + kernel_size , w)
            hstart = max(hstart, 0)
            wstart = max(wstart, 0)
            max_ = 0.0
            i = hstart
            while(i < hend):
                j = wstart
                while(j < wend):
                    #sum += input[i][j]
                    #print(i,j,input[i][j])
                    if input[i][j] > max_:
                        max_ = input[i][j]
                    j+=1
                i+=1
        
            output[ph][pw] = max_
    
    return output

def DPP(input, kernel_size, stride ,padding=0):
    h,w = input.shape
    #print(h,w)
    #floor((W_{in} + 2padding[1] - dilation[1](kernel_size[1] - 1) - 1)/stride[1] + 1
    output_size = np.floor((h + 2*padding - (kernel_size - 1) - 1 )/ stride) + 1
    output_size = int(output_size)
    print(output_size)
    output = np.ones([int(output_size),int(output_size)])*255
    for ph in range(output_size):
        for pw in range(output_size):
            hstart = ph *stride - padding
            wstart = pw * stride -padding
            hend = min(hstart + kernel_size, h)
            #print(kernel_size)
            wend =wstart + kernel_size
            #print(wend)
            if wend - w > 0:
                wend = w
            hstart = max(hstart, 0)
            wstart = max(wstart, 0)
            #max_ = 0.0
            window = []
            i = hstart
            while(i < hend):
                j = wstart
                while(j < wend):
                    #sum += input[i][j]
                    #print(i,j,input[i][j])
                    # if input[i][j] > max_:
                    #     max_ = input[i][j]
                    window.append(input[i][j])
                    j+=1
                i+=1
            ave = sum(window) / len(window)
            rio = (window - ave) 
            rio = np.array(rio)
            rio = (rio**2 ) **(4 / 2) 
            #print(rio)
            ww = rio / sum(rio)
            dpp = [ ww[i]*window[i] for i in range(len(window)) ]
            print(dpp)
            output[ph][pw] = sum(dpp)
    
    return output
    



# max_ = max_pool(input_data,2,2)
# ave = ave_pool(input_data,2,2)
dpp = DPP(input_data,2,1)
print(dpp)
plt.imshow(input_data,cmap=plt.cm.gray)
plt.show()
plt.imshow(dpp,cmap=plt.cm.gray)
plt.show()
