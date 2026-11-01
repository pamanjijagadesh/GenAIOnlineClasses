# Sample data
from langchain_core.documents import Document

docs = [
    Document(page_content="Refund policy: Students can claim a full refund within 14 days "
             "of enrollment. A 50 percent refund is available unitl day 30. No refund"
             "after day 30", metadata={"source": "refund_policy.txt"}),
    Document(page_content="Attendence policy: A minimum of 75 percentage is required "
             "to sit in exams. Medical leave needs a doctor's certificate.",
            metadata={"source": "attendence_policy.txt"}),
    Document(page_content="Library rules: Open 8 AM to 8 PM, Monday to Saturday. Students may "
             "barrow 4 books for 14 days. Late fine is Rs. 2 per day.",
             metadata={"source":"library_rules.txt"}),
    Document(page_content="Hostel rules: Hostel fee is Rs. 60000 per year. Curfew is 10 PM. "
             "Visitors are allowed only on Sundays from 10 AM to 4 PM",
             metadata={"source": "hostels_rules.txt"}),
    Document(page_content="Exam rules: Exams are held twice a year, in December and May. "
             "Students must carry their ID card. Mobile phones are not allowed in the hall.",
             metadata={"source": "exam_rules.txt"})
]