---
published: true
---


I have been looking for a way to control my system GPU behaviour when involved in huge processing tasks. For instance, the current temperature and memory consumption could be good indicators of the GPU state and help prevent possible crashes.

Directly inpired from [nvidia developer site](http://developer.nvidia.com/nvidia-management-library-nvml/),this talk is an attempt to provide the commands mostly used to track and control the GPU parameters.

## 1/- nvidia-smi:
* display GPU general info in a tabular form.

![]({{site.baseurl}}/images/nvidia-smi_0.png)




## 2/- nvidia-smi -L or nvidia-smi --list-gpus
* display a list of GPUs connected to the system.
       
![]({{site.baseurl}}/images/nvidia-smi_1.png)





## 3/- nvidia-smi -q or nvidia-smi --query
* display GPU or unit information.
       
![]({{site.baseurl}}/images/nvidia-smi_2.png)




## 3/a- nvidia-smi -q -u or nvidia-smi -q --unit
* query to display unit rather than the GPU attributes
       
![]({{site.baseurl}}/images/nvidia-smi_3a_.png)




## 3/b- nvidia-smi -q -i or nvidia-smi --id = 0
* Target a specific GPU or Unit.
       
![]({{site.baseurl}}/images/nvidia-smi_3b_.png)


















