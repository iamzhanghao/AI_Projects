a = "()"

def braces(a):
    while a != "":
        a_back = a
        a = a.replace("()", "")
        a = a.replace("{}", "")
        a = a.replace("[]", "")
        if a_back == a:
            print("NO")
            return
    print("YES")


braces(a)