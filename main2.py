##############################################################################################
#################################       Append frames       ##################################
##############################################################################################

    frame_count = 0
    frames_to_process = []
    first = True
    counter = 0
    t1 = time.time()
    while True:
        t2 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == 0 or frame_count % multiple == 0:
            frames_to_process.append(frame)

            if len(frames_to_process) == 8:
                if first:
                    detections = detect(frames_to_process, folder_path, det_model)
                    masks = get_masks(detections, img_size, args.mask_method, args.mask_prior)
                    masks = torch.tensor(masks).unsqueeze(0).float()
                    predictions = test(masks, model)
                    print(f"Prediction: {predictions}")
                    frames_to_process = frames_to_process[1:]
                    first = False
                    print(f"Time for seq: {time.time() - t1:.4} s")
                else:
                    detections = detect(frames_to_process[-1], folder_path, det_model)
                    mask = get_masks(detections, img_size, args.mask_method, args.mask_prior)
                    mask = torch.tensor(mask).unsqueeze(0).float()
                    masks = masks[:,:,1:,:,:]
                    masks = torch.cat((masks, mask), dim=2)
                    predictions = test(masks, model)
                    print(f"Prediction: {predictions}")
                    frames_to_process = frames_to_process[1:]
                    print(f"Time for seq: {time.time() - t2:.4} s")
                
                counter += 1
                print(f"Sequence no.: {counter}\n")
        frame_count += 1
            
        # Break the loop if the 'esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

    print(f"\nVideo Duration: {video_duration} s \nTotal processing time: {time.time() - t1:.4} s")
    cap.release()


    
    # masks = get_masks(detections, img_size, args.mask_method, args.mask_prior)
    # print(masks.shape)
    # masks = torch.tensor(masks).unsqueeze(0).float()
    # print(f"mask shape in main: {masks.shape}")
    # print(f"Data type of mask: {masks.type()}")
    
    # python main.py --run_name Thesis_test --input mask --backbone ResNext18 --mask_method "case4"
    
print("-"*79, "\n", "-"*79, "\n" * 5)


##############################################################################################
#################################     Append detections     ##################################
##############################################################################################

    frame_count = 0
    detections = []
    first = True
    counter = 0
    t1 = time.time()
    while True:
        t2 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == 0 or frame_count % multiple == 0:
            if first:
                detections.append(detect(frame, folder_path, det_model)[0])
            else:
                dets = detect(frame, folder_path, det_model)

            if len(detections) == 8:
                if first:
                    masks = get_masks(detections, img_size, args.mask_method, args.mask_prior)
                    masks = torch.tensor(masks).unsqueeze(0).float()
                    predictions = test(masks, model)
                    print(f"Prediction: {predictions}")
                    first = False
                    print(f"Time for the first sequence (8 detections + 1 prediction): {time.time() - t1:.4} s")
                else:
                    mask = get_masks(dets, img_size, args.mask_method, args.mask_prior)
                    mask = torch.tensor(mask).unsqueeze(0).float()
                    masks = masks[:,:,1:,:,:]
                    masks = torch.cat((masks, mask), dim=2)
                    predictions = test(masks, model)
                    print(f"Prediction: {predictions}")
                print(f"Time for seq: {time.time() - t2:.4} s")    
                
                counter += 1
                print(f"Sequence no.: {counter}")
        frame_count += 1

    print(f"\nVideo Duration: {video_duration} s \nTotal processing time: {time.time() - t1:.4} s")
    cap.release()

##############################################################################################
################################# Append detections + masks ##################################
##############################################################################################

    frame_count = 0
    first = True
    counter = 0
    t1 = time.time()
    while True:
        t2 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == 0 or frame_count % multiple == 0:
            mask = get_masks(detect(frame, folder_path, det_model), img_size, args.mask_method, args.mask_prior)
            mask = torch.tensor(mask).unsqueeze(0).float()
            if first:
                masks = mask
                first = False
            else:
                masks = torch.cat((masks, mask), dim=2)

            if masks.shape[2] == 8:
                predictions = test(masks, model)
                print(f"Prediction: {predictions}")
                masks = masks[:,:,1:,:,:]
                print(f"Time for seq: {time.time() - t2:.4} s")    
                counter += 1
                print(f"Sequence no.: {counter}")
                if counter == 1:
                    print(f"Time for the first seq (8 detections + 1 prediction): {time.time() - t1:.4} s")
        
                # Plot the prediction for each frame
                # frame = cv2.putText(frame, f"Prediction: {predictions[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.imshow('Frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        frame_count += 1

    print(f"\nVideo Duration: {video_duration} s \nTotal processing time: {time.time() - t1:.4} s")
    cap.release()
    cv2.destroyAllWindows()