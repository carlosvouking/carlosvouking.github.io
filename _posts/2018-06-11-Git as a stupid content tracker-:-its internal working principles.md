---
published: true
---

[Wikipedia](https://en.wikipedia.org/wiki/Git) says that **'Git is a distributed revision control system'**. It tracks files and directories accross computer systems(_distribution_). Tracking and interacting with those files and directories require the use of specific commands in a bash terminal.

Git appears to be like a mapping table which links **keys** and **values** and save the (key, values) pairs in a system. To make it simple, let's say I am writing my thesis and as it goes, I would like to gradually edit it and save every edited copy of it. Each edited copy is mapped to a specific unique key and stored in a table. Git is just that table. Technically, the text in my thesis report is called a **body** and represents a sequence of bites. Giving a body to the mapping table (that is git),  generates a unique key also called a **hash**. The hash is computed using an algorithm called **SHA1** (Secure hashing algorithm). Each time I add modification to my thesis report and submit it to Git, it takes it, creates a new hash, couples it to the body and persist the **(hash, body)** pair in the system. there is only and exactly one hash for each body string.

The benefit of this is that, at a certain point in time, I can ask git to give me a specific saved version of my thesis report and it will be able to give me the exact one based on its  corresponding hash. That is conceptually how git works internally.

**Let's loo at it that practically.**
If I consider passing the string _"deep learning"_ to git, it will create the following hash key:  _'b075a46024e2ea2b418a26ea14d4a57759fbf3d1'_.
By firing up your git bash commnad terminal and typing: '$ echo 'deep learning' | git hash-object --stdin':

![png](/images/git1.PNG)