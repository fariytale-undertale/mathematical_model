#大小写
name="ada lovelace"
print(name.title())#首字母
print(name.upper())#全部
print(name.lower())#全部小写
#在存储数据时，可将输入全部转化为小写，再按需转换成合适的大小写方式

first_name="ada"
last_name="lovelace"
full_name=f"{first_name} {last_name}"#f将每个变量都替换成其值
print(full_name)
print(f"Hello,{full_name.title()}!")

#删除字符串的空白，包括空格，\n,\t，（）内可指出想要去除的字符，返回一个新字符串
favourite_language=" python "
favourite_language=favourite_language.rstrip()#删除右空格
favourite_language=favourite_language.lstrip()#删除左空格
favourite_language=favourite_language.strip()#删除两端空格

#删除前缀
nostarch_url="http://nostarch.com"
print(nostarch_url.removeprefix("http://"))
print(nostarch_url.removesuffix(".com"))