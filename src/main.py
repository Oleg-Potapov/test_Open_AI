from fastapi import FastAPI, HTTPException
import uvicorn
from src.config import OPENAI_API_KEY
from src.schemas import QuestionRequest
from src.services.llm_service import LlmService, urls

app = FastAPI()


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    try:
        llm_service = LlmService(urls, openai_api_key=OPENAI_API_KEY)
        answer = llm_service.answer_question(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке вопроса: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
