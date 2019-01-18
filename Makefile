TARGETS=README.md

.PHONY: all clean

all: $(TARGETS)

clean:
	$(RM) $(TARGETS)

README.md: doc/README_math.md
	cat $< | python ./scripts/md_replace_math.py > $@

