TARGETS=README.md

.PHONY: all clean

all: $(TARGETS)

clean:
	$(RM) $(TARGETS)

README.md: doc/README_math.md ./scripts/md_replace_math.py
	cat $< | python ./scripts/md_replace_math.py --image_directory doc/img > $@

