from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    
    name:str
    age:Optional[int]
    email: EmailStr
    cgpa: float =Field(gt=0, lt=10)
    
new_student = {'name':'Khurram','age':'44','email':'jawadansar556@gmail.com','cgpa':'7'}

Student = Student(**new_student)

print(Student)