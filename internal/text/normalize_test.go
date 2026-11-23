package text

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"encoding/json"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/fixtures"
)

// spokenCase is an input and its spoken form.
type spokenCase struct {
	in, want string
}

// The expected texts are the training data's spoken forms of each input.
func TestSpokenText(t *testing.T) {
	for _, c := range []spokenCase{
		{"Hello there.", "Hello there."},
		{"It costs $5.50 today.", "It costs five dollars and fifty cents today."},
		{"$1 and \u00a32", "one dollar and two pounds"},
		{"$3 million and $1.5M", "three million dollars and one point five million dollars"},
		{"In 1999 and 2005 and 2024.", "In nineteen ninety nine and two thousand five and twenty twenty four."},
		{"The 1st, 22nd and 103rd.", "The first, twenty second and one hundred and third."},
		{"1,234,567 people", "one million two hundred and thirty four thousand five hundred and sixty seven people"},
		{"50% off, 3.14 pi", "fifty percent off, three point one four pi"},
		{"Meet at 5 p.m. Then go.", "Meet at five PM. Then go."},
		{"The webinar starts at 4 p.m. ET.", "The webinar starts at four PM ET."},
		{"At 14:00 and 08:30.", "At fourteen hundred hours and oh eight thirty."},
		{"Caf\u00e9 \u201cquoted\u201d \u2014 done", "Cafe \"quoted\" - done"},
		{"Wait--what? He said --)", "Wait - what? He said -)"},
		{"Fine <laugh>  really , yes", "Fine <laugh> really, yes"},
		{"Call 555-0199 or (555) 123-4567.",
			"Call five five five, zero one nine nine or five five five, one two three, four five six seven."},
		{"Email john.doe@example.com or visit https://example.com/docs?id=3.",
			"Email john dot doe at example dot com or visit example dot com slash docs."},
		{"Save it as notes_v2.txt <laugh> now.", "Save it as notes underscore V two dot T X T <laugh> now."},
		{"Please update to v2.4.1 first.", "Please update to version two point four point one first."},
		{"Chapter IV tells how Henry VIII ruled.", "Chapter four tells how Henry the eighth ruled."},
		{"Dr. Brown lives on Main St. The house is red.", "Doctor Brown lives on Main Street. The house is red."},
		{"Boston, MA 02115", "Boston, Massachusetts zero two one one five"},
		{"It was 72\u00b0F and -3.5\u00b0C.",
			"It was seventy two degrees Fahrenheit and minus three point five degrees Celsius."},
		{"a 5 kg bag and 12 oz. of flour", "a five kilogram bag and twelve ounces of flour"},
		{"3/4 of the class and on 3/14", "three quarters of the class and on March fourteenth"},
		{"Shipped on 23/09/2026.", "Shipped on the twenty third of September twenty twenty six."},
		{"the 1990s and '80s", "the nineteen nineties and eighties"},
		{"Order No. 7741890", "Order Number seven seven four one eight nine zero"},
		{"#ThrowbackThursday with @sales_team", "hashtag Throwback Thursday with at sales team"},
		{"git rebase -i and ls -la", "git rebase dash I and ls dash L A"},
		{"Ctrl+Shift+T", "control shift T"},
		{"btw can u grab it tmrw", "by the way can you grab it tomorrow"},
		{"## Setup\n- Fast setup\n1. Open **Settings**", "Setup. Fast setup. one, Open Settings."},
		{"Here are the steps: 1. Open Settings 2. Tap Wi-Fi 3. Turn it on.",
			"Here are the steps: one, Open Settings. two, Tap Wi-Fi. three, Turn it on."},
		{"I have 2. Then version 2. It works.", "I have two. Then version two. It works."},
		{"> quoted\n>> nested", "quoted. nested."},
		{"She said:\n> > it works\nThanks.", "She said: it works. Thanks."},
		{"> - Fast setup\n> 1. Open it.", "Fast setup. one, Open it."},
		{"> 5 years needed", "greater than five years needed"},
		{"1. Open 2. Tap\nNext line", "one, Open. two, Tap. Next line"},
		{"Thanks :) See you", "Thanks. See you"},
		{"(great :)) See you", "(great). See you"},
		{"5 > 3 and x>y", "five greater than three and x is greater than y"},
		{"cat :) lol", "cat lol"},
		{"(great :)) then", "(great) then"},
		{"Note:) see", "Note:) see"},
		{"See [the docs](https://x.io).", "See the docs."},
		{"x\u00b2 + y\u00b3 = 10\u2076", "x squared plus y cubed equals ten to the sixth"},
		{"\u00bd cup and 1\u00bd cups", "one half cup and one and a half cups"},
		{"na\u00efve Zo\u00eb e\u0301", "naive Zoe e"},
		{"1234567890123456789012", "one two three four five six seven eight nine zero one two three four five six " +
			"seven eight nine zero one two"},
		{"Rock & roll", "Rock and roll"},
		{"", ""},
		{"<laugh>", "<laugh>"},
	} {
		if got, err := Spoken(c.in); err != nil || got != c.want {
			t.Errorf("Spoken(%q) = %q, %v; want %q", c.in, got, err, c.want)
		}
	}

	if got, err := Spoken("\u65e5\u672c"); err == nil {
		t.Errorf("Spoken kept characters the model cannot say: %q", got)
	}
}

// normalizeGoldenRow is one input of the normalizer corpus with the training data's text for it and whether that text
// is spoken.
type normalizeGoldenRow struct {
	In     string `json:"in"`
	Out    string `json:"out"`
	Spoken bool   `json:"spoken"`
}

// TestSpokenTextGolden compares Spoken with the training data's spoken forms on the corpus TWOCENT_NORMALIZE_GOLDEN
// names: where that text is spoken Spoken must return it byte for byte, and where it is not Spoken must refuse,
// except on the inputs testdata/accepted.txt lists.
func TestSpokenTextGolden(t *testing.T) {
	acceptedDifferences, err := readAccepted(acceptedText)
	if err != nil {
		t.Fatal(err)
	}

	rows, err := readNormalizeGolden(fixtures.LoadNormalizeGolden(t))
	if err != nil {
		t.Fatal(err)
	}

	if len(rows) == 0 {
		t.Fatal("the normalizer corpus holds no rows")
	}

	results := spokenTextResults(rows)
	mismatches, accepted := 0, 0
	for i, row := range rows {
		got := results[i]
		if row.Spoken == (got.err == nil) && (!row.Spoken || got.text == row.Out) {
			continue
		}

		if acceptedDifferences[row.In] {
			accepted++
			continue
		}

		mismatches++
		if mismatches > 40 {
			continue
		}

		if row.Spoken {
			t.Errorf("Spoken(%q)\n got  %q, %v\n want %q", row.In, got.text, got.err, row.Out)
		} else {
			t.Errorf("Spoken(%q) = %q, want a refusal of %q", row.In, got.text, row.Out)
		}
	}

	t.Logf("%d of %d texts mismatch; %d accepted differences", mismatches, len(rows), accepted)
}

//go:embed testdata/accepted.txt
var acceptedText string

// readAccepted reads the accepted inputs: one Go quoted string per line, with blank lines and # comments between.
func readAccepted(text string) (map[string]bool, error) {
	inputs := map[string]bool{}
	for i, line := range strings.Split(text, "\n") {
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		input, err := strconv.Unquote(line)
		if err != nil {
			return nil, fmt.Errorf("accepted.txt line %d: %w", i+1, err)
		}
		inputs[input] = true
	}
	return inputs, nil
}

func readNormalizeGolden(data []byte) ([]normalizeGoldenRow, error) {
	r, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	var rows []normalizeGoldenRow
	err = json.NewDecoder(r).Decode(&rows)
	return rows, err
}

// spokenTextResult is Spoken's text and error for one input.
type spokenTextResult struct {
	text string
	err  error
}

// spokenTextResults runs Spoken on every row, spread over all CPUs.
func spokenTextResults(rows []normalizeGoldenRow) []spokenTextResult {
	results := make([]spokenTextResult, len(rows))
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	for worker := range workers {
		wg.Go(func() {
			for i := worker; i < len(rows); i += workers {
				text, err := Spoken(rows[i].In)
				results[i] = spokenTextResult{text, err}
			}
		})
	}
	wg.Wait()
	return results
}
