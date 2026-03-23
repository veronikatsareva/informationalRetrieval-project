from flask import Flask, render_template, request
import index
import time

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("main.html")


@app.route("/search")
def search():
    return render_template("search.html")


@app.route("/results", methods=["POST"])
def results():
    start = time.time()
    query, indexType, rankNum = (
        request.form.get("searchField"),
        request.form.get("indexType"),
        request.form.get("rankNum"),
    )
    if not query:
        return render_template("search.html")

    queryChecked, records = index.search(query, indexType, rankNum)
    end = time.time()

    return render_template(
        "results.html",
        queryChecked=queryChecked,
        indexType=indexType,
        time=end - start,
        N=len(records),
        records=records,
    )


if __name__ == "__main__":
    app.run(debug=True)
