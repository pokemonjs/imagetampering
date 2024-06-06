def fun(fn="data/list_casia1_splice.txt",str1="data/Columbia",str2="data/CASIA1"):
	p=open(fn).readlines()
	arr=[]
	for pp in p:
		arr.append(pp.replace(str1,str2))
	with open(fn,"w") as f:
		for p in arr:
			f.write(p)

# fun()
# fun("data/train_list_san.txt","data/data163850","data/SAN")
# fun("data/val_list_san.txt","data/data163850","data/SAN")
fun("data/list_columbia.txt","data/Columbia/4cam_splc/4cam_splc","data/Columbia")