from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text ="""class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def check_result(self):
        if self.marks >= 50:
            print(self.name, "has passed")
        else:
            print(self.name, "has failed")


student1 = Student("Ali", 75)
student1.check_result()"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0],'\n',chunks[1])