from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

if __name__ == "__main__":
    outputs = []
    #print(app.invoke(input={"question": "If the ball hits defenders hand from his body part in close range, does it penalty?"}).values())
    for s in app.stream(input={"question": "which situations are penalised with red card in football?"}, stream_mode="values"):
        outputs.append(s)

    print(outputs[-1]["generation"])