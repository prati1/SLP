with open ("lspd.txt","r",encoding="utf-8") as f:
  corpus=f.read()
  
  
char=sorted(list(set(corpus)))

encoding={c:i for i,c in enumerate(char)}
decoding={i:c for i,c in enumerate(char)}

print(encoding)
print(decoding)
print("Unique characters "+str(len(char)))
print("Total characters "+str(len(corpus)))
import io
with io.open("char.txt", "w", encoding="utf-8") as f:
    f.write(str(encoding) + "\nUnique characters "+str(len(char))+ "\nTotal characters "+str(len(corpus)))
	#f.write(decoding)