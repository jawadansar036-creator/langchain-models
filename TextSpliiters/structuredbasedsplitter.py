from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """The poem celebrates the excitement, energy, and joy of playing cricket. It describes a cricket match on a bright and sunny day, where players step onto the field with confidence and enthusiasm.
The batsman holds the bat and prepares to face the bowler, while the bowler runs toward the wicket and delivers the ball.

The poem captures the excitement of the game when the batsman hits the ball and the crowd cheers loudly. Hitting a six is shown as a particularly thrilling moment. 
The players run between the wickets to score runs and work hard to increase their team's score. At the same time, wickets may fall, but the players do not lose hope or courage."""

splitter =RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)