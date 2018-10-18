---
published: true
---


I have been looking for a way to control my system GPU behaviour when involved in huge processing tasks. For instance, the current temperature and memory consumption could be good indicators of the GPU state and help prevent possible crashes.

Directly inpired from [nvidia developer site](http://developer.nvidia.com/nvidia-management-library-nvml/),this talk is an attempt to provide the commands mostly used to track and control the GPU parameters.

## I/- nvidia-smi:

![]({{site.baseurl}}/images/nvidia-smi_.png)


## I/- nvidia-smi -L or nvidia-smi --list-gpus
       display a list of GPUs connected to the system.
       
![]({{site.baseurl}}/images/nvidia-smi_1.png)       
