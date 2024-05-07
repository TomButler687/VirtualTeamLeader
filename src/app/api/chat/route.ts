import { Message as VercelChatMessage, StreamingTextResponse, createStreamDataTransformer } from 'ai';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { HttpResponseOutputParser } from 'langchain/output_parsers';
import { MongoDBAtlasVectorSearch } from '@langchain/mongodb';
import { MongoClient } from "mongodb";
import { RunnableSequence } from '@langchain/core/runnables'


const uri = process.env.OPEN_AI_KEY!

const TEMPLATE = `This is all in relation to customer travel insurance. We are the insurance provider. You are to give me relevant information from the context passed to you in order for me to answer a customers questions. Take this into consideration when replying. Always give examples and evidence where possible.
Answer in specific detail instead of general advice.
Answer the user's questions based on the provided context. If the answer is not in the context, reply politely that you do not have that information available.

Context: {context}

Current conversation: {chat_history}
user: {question}
assistant:`;

export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  try {
    // Extract the `messages` from the body of the request
    const { messages } = await req.json();
    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
    const currentMessageContent = messages[messages.length - 1].content;

    const client = new MongoClient(uri);
    const database = client.db("testing")
    const collection = database.collection("testing-embed")
    const dbConfig = {
        collection: collection,
        indexName: "default",
        textKey: "text", // Field name for the raw text content. Defaults to "text".
        embeddingKey: "embedding"
    }

    // Connect to your MongoDB Atlas database and load the vector store
    const vectorStore = new MongoDBAtlasVectorSearch(new OpenAIEmbeddings(), dbConfig)

    const prompt = PromptTemplate.fromTemplate(TEMPLATE);
    const model = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-3.5-turbo',
      temperature: 0,
      streaming: true,
      verbose: true,
    });

    const parser = new HttpResponseOutputParser();


    const basicOutput = await vectorStore.similaritySearch(currentMessageContent);
    const basicResults = basicOutput.map((results => ({
    pageContent: results.pageContent,
    })))

    const chain = RunnableSequence.from([
        {
            question: (input) => input.question,
            chat_history: (input) => input.chat_history,
            context: () => basicResults.map(result => result["pageContent"]).join("\n"),
        },
        prompt,
        model,
        parser,
    ]);

    // Convert the response into a friendly text-stream
    const stream = await chain.stream({
        chat_history: formattedPreviousMessages.join('\n'),
        question: currentMessageContent,
    });

    // Respond with the stream
    return new StreamingTextResponse(
      stream.pipeThrough(createStreamDataTransformer()),
  );

  } catch (e: any) {
    return Response.json({ error: e.message }, { status: e.status ?? 500 });
  }
}

const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};