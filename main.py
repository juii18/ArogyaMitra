from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

app = FastAPI(title="ArogyaMitra API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ── Models ──────────────────────────────────────────────────────────────────

class WorkoutRequest(BaseModel):
    age: int
    gender: str
    goal: str  # Weight Loss, Muscle Gain, Flexibility, etc.
    workout_type: str  # Home, Gym, Outdoor
    daily_minutes: int
    fitness_level: str  # Beginner, Intermediate, Advanced
    health_conditions: Optional[str] = ""

class NutritionRequest(BaseModel):
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    goal: str
    calorie_target: int
    diet_type: str  # Vegetarian, Vegan, Non-Veg
    allergies: Optional[str] = ""
    cuisine_preference: Optional[str] = "Indian"

class AROMIMessage(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []
    user_profile: Optional[dict] = {}

class ProgressRequest(BaseModel):
    user_name: str
    workouts_completed: int
    streak_days: int
    calories_burned: int
    goal: str

# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ArogyaMitra API is running 🌿"}


@app.post("/api/workout-plan")
async def generate_workout_plan(req: WorkoutRequest):
    prompt = f"""You are ArogyaMitra, an expert AI fitness coach. Generate a complete personalized 7-day workout plan.

User Profile:
- Age: {req.age} years
- Gender: {req.gender}
- Goal: {req.goal}
- Workout Type: {req.workout_type}
- Available Time: {req.daily_minutes} minutes/day
- Fitness Level: {req.fitness_level}
- Health Conditions: {req.health_conditions or 'None'}

Generate a detailed 7-day workout plan in JSON format with this EXACT structure:
{{
  "overview": "Brief motivating overview of the plan",
  "weekly_goal": "Specific weekly target",
  "days": [
    {{
      "day": 1,
      "day_name": "Monday",
      "focus": "Focus area (e.g., Full Body, Cardio)",
      "duration_minutes": {req.daily_minutes},
      "warmup": {{
        "duration_minutes": 5,
        "exercises": ["exercise1", "exercise2"]
      }},
      "main_workout": [
        {{
          "name": "Exercise Name",
          "sets": 3,
          "reps": "12-15",
          "rest_seconds": 60,
          "description": "How to do it",
          "difficulty": "Easy/Medium/Hard"
        }}
      ],
      "cooldown": {{
        "duration_minutes": 5,
        "exercises": ["stretch1", "stretch2"]
      }},
      "youtube_search": "search query for YouTube",
      "tip": "Daily fitness tip",
      "calories_estimate": 250
    }}
  ],
  "weekly_tips": ["tip1", "tip2", "tip3"],
  "progression_advice": "How to progress after this week"
}}

Make it realistic, safe, and motivating. Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
        )
        import json
        raw = response.choices[0].message.content.strip()
        # Clean markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return {"success": True, "plan": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/nutrition-plan")
async def generate_nutrition_plan(req: NutritionRequest):
    prompt = f"""You are ArogyaMitra, an expert AI nutritionist. Generate a complete 7-day meal plan.

User Profile:
- Age: {req.age} | Gender: {req.gender}
- Weight: {req.weight_kg}kg | Height: {req.height_cm}cm
- Goal: {req.goal}
- Daily Calorie Target: {req.calorie_target} kcal
- Diet Type: {req.diet_type}
- Allergies/Restrictions: {req.allergies or 'None'}
- Cuisine Preference: {req.cuisine_preference}

Generate a detailed 7-day meal plan in JSON format:
{{
  "overview": "Nutrition philosophy for this plan",
  "daily_macros": {{
    "protein_g": 120,
    "carbs_g": 180,
    "fat_g": 60,
    "fiber_g": 30
  }},
  "days": [
    {{
      "day": 1,
      "day_name": "Monday",
      "total_calories": {req.calorie_target},
      "meals": {{
        "breakfast": {{
          "name": "Meal name",
          "items": ["item1 - quantity", "item2 - quantity"],
          "calories": 400,
          "protein_g": 20,
          "carbs_g": 50,
          "fat_g": 12,
          "prep_time_mins": 10,
          "recipe_tip": "Quick tip"
        }},
        "morning_snack": {{
          "name": "Snack name",
          "items": ["item1"],
          "calories": 150,
          "protein_g": 5,
          "carbs_g": 20,
          "fat_g": 5,
          "prep_time_mins": 2,
          "recipe_tip": "tip"
        }},
        "lunch": {{
          "name": "Meal name",
          "items": ["item1", "item2"],
          "calories": 550,
          "protein_g": 30,
          "carbs_g": 60,
          "fat_g": 18,
          "prep_time_mins": 20,
          "recipe_tip": "tip"
        }},
        "evening_snack": {{
          "name": "Snack",
          "items": ["item"],
          "calories": 150,
          "protein_g": 8,
          "carbs_g": 15,
          "fat_g": 5,
          "prep_time_mins": 5,
          "recipe_tip": "tip"
        }},
        "dinner": {{
          "name": "Meal name",
          "items": ["item1", "item2"],
          "calories": 550,
          "protein_g": 35,
          "carbs_g": 55,
          "fat_g": 20,
          "prep_time_mins": 25,
          "recipe_tip": "tip"
        }}
      }},
      "hydration_tip": "Daily water/hydration advice",
      "wellness_note": "Additional wellness tip"
    }}
  ],
  "grocery_list": {{
    "proteins": ["item1", "item2"],
    "vegetables": ["item1", "item2"],
    "grains": ["item1"],
    "fruits": ["item1"],
    "dairy_alternatives": ["item1"],
    "spices": ["item1"]
  }},
  "supplement_advice": "Optional supplement suggestions"
}}

Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=5000,
        )
        import json
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return {"success": True, "plan": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aromi-chat")
async def aromi_chat(req: AROMIMessage):
    system_prompt = """You are AROMI, the friendly and motivating AI wellness coach of ArogyaMitra platform. 

Your personality:
- Warm, encouraging, and empathetic
- Expert in fitness, nutrition, yoga, and wellness
- Adaptive - you adjust plans based on user's situation
- Use occasional fitness/wellness emojis
- Give practical, actionable advice
- Reference Indian wellness traditions (Ayurveda, yoga) when relevant
- Keep responses concise but thorough (150-300 words)
- Always end with a motivating line or actionable next step

You help with:
- Adjusting workout plans for travel, injury, mood
- Nutrition guidance and meal suggestions  
- Motivation and mental wellness
- Sleep and recovery tips
- Hydration and supplement advice
- Yoga and mindfulness recommendations

User Profile Context:
""" + str(req.user_profile)

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for msg in req.conversation_history[-10:]:  # Last 10 messages for context
        messages.append(msg)
    
    messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=600,
        )
        reply = response.choices[0].message.content
        return {"success": True, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/progress-insights")
async def progress_insights(req: ProgressRequest):
    prompt = f"""You are AROMI from ArogyaMitra. Generate personalized progress insights and motivation.

User: {req.user_name}
Workouts Completed: {req.workouts_completed}
Current Streak: {req.streak_days} days
Calories Burned: {req.calories_burned} kcal
Goal: {req.goal}

Respond in JSON:
{{
  "headline": "Exciting personalized headline",
  "progress_score": 78,
  "badges": ["badge1", "badge2"],
  "insights": ["insight1", "insight2", "insight3"],
  "next_milestone": "What to aim for next",
  "charity_contribution": "₹{req.workouts_completed * 10} donated to health causes",
  "motivational_quote": "Personalized quote",
  "recommendations": ["rec1", "rec2"]
}}

Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        import json
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return {"success": True, "insights": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)