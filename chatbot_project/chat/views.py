from django.shortcuts import render
from django.http import JsonResponse
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load biến môi trường từ .env
load_dotenv()

# Cấu hình Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
generative_model = genai.GenerativeModel("gemini-2.0-flash")

# Load dữ liệu
data_df = pd.read_csv("data.csv")  # Gồm: questions, answers, questions_processed

# Load model nhúng tiếng Việt
embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert")  # Hoặc model phù hợp khác

# Tính embedding cho toàn bộ câu hỏi khi server khởi động
questions_list = data_df["questions"].tolist()
question_embeddings = embedding_model.encode(questions_list)

def index(request):
    return render(request, "chat/index.html")

def get_response(request):
    user_message = request.GET.get("message", "").strip()

    if not user_message:
        return JsonResponse({"response": "Vui lòng nhập câu hỏi."})

    try:
        # Chuyển câu hỏi người dùng thành vector
        user_embedding = embedding_model.encode([user_message])

        # Tính độ tương đồng cosine giữa câu hỏi và các embeddings trong cơ sở dữ liệu
        similarities = cosine_similarity(user_embedding, question_embeddings)[0]
        top_idx = similarities.argmax()
        similarity_score = similarities[top_idx]

        # Ngưỡng độ tương đồng để quyết định trả lời
        threshold = 0.6  # Có thể điều chỉnh tùy nhu cầu

        if similarity_score < threshold:
            content = "Xin lỗi, tôi chưa có thông tin phù hợp để trả lời câu hỏi này."
        else:
            matched_question = data_df.iloc[top_idx]["questions"]
            matched_answer = data_df.iloc[top_idx]["answers"]

            prompt = f"""
            Dựa trên dữ liệu thực sau đây:

            Câu hỏi gốc trong cơ sở dữ liệu: "{matched_question}"
            Câu trả lời: "{matched_answer}"

            Người dùng hỏi: "{user_message}"

            Hãy trả lời ngắn gọn, rõ ràng, dễ hiểu theo ngữ cảnh tiếng Việt.
            """

            response = generative_model.generate_content(prompt)
            content = response.text.strip()

    except Exception as e:
        content = f"Đã xảy ra lỗi: {str(e)}"

    return JsonResponse({"response": content})
