import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";

dotenv.config();

// Lazily initialize Gemini AI client
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is not set. Please configure it in the Secrets panel.");
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Body parser limit increased slightly for reports/chat history
  app.use(express.json());

  // === AI API Endpoints ===

  // 1. Personalized Quiz Generator
  app.post("/api/quiz", async (req, res) => {
    try {
      const { subject, topic, age, qcount, difficulty } = req.body;
      if (!subject || !topic) {
        return res.status(400).json({ error: "Subject and topic are required." });
      }

      const client = getGeminiClient();

      const prompt = `You are an expert school-level teacher and educational content creator. Create a multiple-choice quiz about "${topic}" for a ${age}-year-old student studying ${subject}.
The quiz must have exactly ${qcount || 4} questions.
The difficulty of target questions must be tailored to "${difficulty || 'Easy'}" level for a child of age ${age}.

Please make the multiple choice questions highly interactive, engaging, and age-appropriate. Make sure the explanation is positive and educational.`;

      const response = await client.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: {
                type: Type.STRING,
                description: "Catchy and encouraging quiz title matching topic and age level."
              },
              questions: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    question: { type: Type.STRING, description: "The actual quiz question." },
                    options: {
                      type: Type.ARRAY,
                      items: { type: Type.STRING },
                      description: "Exactly 4 options, labeled like 'A) ...', 'B) ...', 'C) ...', 'D) ...'."
                    },
                    correct: {
                      type: Type.INTEGER,
                      description: "The 0-based index of the correct option (0, 1, 2, or 3)."
                    },
                    explanation: {
                      type: Type.STRING,
                      description: "A friendly, encouraging explanation of why this selection is correct and how to understand it."
                    }
                  },
                  required: ["question", "options", "correct", "explanation"]
                }
              }
            },
            required: ["title", "questions"]
          }
        }
      });

      const responseText = response.text;
      if (!responseText) {
        throw new Error("Received empty response from Gemini");
      }

      const quizData = JSON.parse(responseText.trim());
      res.json(quizData);
    } catch (error: any) {
      console.error("Quiz Generation Error:", error);
      res.status(500).json({ error: error.message || "Failed to generate quiz. Please check backend logs." });
    }
  });

  // 2. Interactive AI Tutor Chat Gateway
  app.post("/api/tutor", async (req, res) => {
    try {
      const { messages, subject } = req.body;
      if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({ error: "Valid chat messages array is required." });
      }

      const client = getGeminiClient();

      // Convert client message format into Gemini client SDK parts format
      const contents = messages.map((m: any) => {
        const role = m.role === "assistant" ? ("model" as const) : ("user" as const);
        const parts: any[] = [];

        if (m.attachment && m.attachment.base64) {
          // Send the media attachment directly in Gemini's multimodal layout
          parts.push({
            inlineData: {
              mimeType: m.attachment.type,
              data: m.attachment.base64
            }
          });
          // Guide the model with text so it acknowledges the attached file
          parts.push({
            text: `[Student uploaded file: "${m.attachment.name}" of type ${m.attachment.type}] ${m.content || "Please look at my uploaded attachment and answer."}`
          });
        } else {
          parts.push({ text: m.content || "" });
        }

        return { role, parts };
      });

      const systemInstruction = `You are a friendly, encouraging, and highly knowledgeable AI tutor for school students aged 10–15.
The student is currently studying the subject: ${subject || "Science"}.

Your behavior principles:
- Explain subject-matter concepts clearly with simple, age-appropriate language (avoid overly dense academic jargon).
- Use relatable, memorable real-world analogies and examples.
- Break down complex mathematical, scientific, physical, or historical explanations into clean, numbered steps.
- Be warm, supportive, and extremely encouraging. If the student makes an error or doesn't understand, guide them with patience.
- Conclude responses with a short question or prompt to check understanding and keep the dialog interactive.
- Limit explanations to a readable length (ideally 3-6 sentences, or clear listed bullet-points for complex queries).

VISUAL TUTOR RULE (CRITICAL - ALWAYS FOLLOW):
- Since children are visual learners, for every educational topic explanation (unless it is just a simple greeting/acknowledgement), you MUST include exactly 2 or 3 beautiful illustration or photography images in your explanation to teach the concepts.
- Choose highly relevant keywords and use standard Markdown syntax for these images:
  ![Detailed description of what this picture/diagram represents](https://images.unsplash.com/featured/600x400/?<educational_keyword>&sig=<index>)
- Replace '<educational_keyword>' with a highly specific keyword related to the concept or element you are teaching (e.g. "telescope", "volcano", "pyramid", "oxygen", "fractions").
- Replace '<index>' with a unique number (like 1, 2, or 3) for each image to prevent browser duplicate image caching collisions.
- Distribute the images nicely inside your text (e.g., between paragraphs) and write brief direct descriptions referenceable to these images for a great child-friendly layout.`;

      const response = await client.models.generateContent({
        model: "gemini-3.5-flash",
        contents: contents,
        config: {
          systemInstruction,
          temperature: 0.7,
        }
      });

      const reply = response.text || "I'm sorry, I could not gather a proper answer. Let's try rephrasing your topic!";
      res.json({ reply });
    } catch (error: any) {
      console.error("AI Tutor Error:", error);
      res.status(500).json({ error: error.message || "Tutor request failed." });
    }
  });

  // 3. AI Progress Report Creator
  app.post("/api/report", async (req, res) => {
    try {
      const { name, scores, notes } = req.body;
      if (!name) {
        return res.status(400).json({ error: "Student full name is required." });
      }

      const client = getGeminiClient();

      const prompt = `You are an experienced, highly empathetic school teacher compiling a professional academic progress report.
Analyze the student metrics and provide constructive observations. Keep the tone warm, actionable, and encouraging for parents.

Student Name: ${name}
Scores Table/Details:
${scores || "No direct scores available."}

Teacher Observations / Extra Notes:
${notes || "No extra commentary."}`;

      const response = await client.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              summary: {
                type: Type.STRING,
                description: "A professional, balanced 2-3 sentence overview of academic standing."
              },
              strengths: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: "List 3 specific areas of strength or excellent behavior."
              },
              areas_for_growth: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: "List 2 concrete, supportive points of subject-specific areas needing improvement/practice."
              },
              recommendation: {
                type: Type.STRING,
                description: "An actionable 1-2 sentence recommendation for parents/guardians to support learning at home."
              },
              teacher_comment: {
                type: Type.STRING,
                description: "A highly personal, encouraging 3-4 sentence message written directly to the student (${name}) validating their style and motivating them for future terms."
              }
            },
            required: ["summary", "strengths", "areas_for_growth", "recommendation", "teacher_comment"]
          }
        }
      });

      const responseText = response.text;
      if (!responseText) {
        throw new Error("Received empty report from Gemini");
      }

      const reportData = JSON.parse(responseText.trim());
      res.json(reportData);
    } catch (error: any) {
      console.error("Report Generation Error:", error);
      res.status(500).json({ error: error.message || "Failed to generate student report." });
    }
  });

  // === Vite & Frontend Integration ===

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server fully active at http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Server failure to boot:", err);
  process.exit(1);
});
