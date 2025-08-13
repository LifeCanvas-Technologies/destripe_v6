    with tqdm(total=len(args), gui=True, bar=gui['tile_progress']) as pbar:
        with multiprocessing.Pool(workers) as pool:
            for result in pool.imap(_read_filter_save, args, chunksize=chunks):
                pbar.update()
                pbar.refresh()