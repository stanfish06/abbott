%%
img_r1 = double(tiffreadVolume("./chik/stitched_p0000_w0000_t0000_R1.tif"));
img_r2 = double(tiffreadVolume("./chik/stitched_p0000_w0000_t0000_R2.tif"));
img_r2_reg = double(tiffreadVolume("../registered_img.tif"));
%%
img_r1 = img_r1 / max(img_r1(:));
img_r2 = img_r2 / max(img_r2(:));
img_r2_reg = img_r2_reg / max(img_r2_reg(:));
%%
mip_r1 = imadjust(max(img_r1, [], 3));
mip_r2 = imadjust(max(img_r2, [], 3));
mip_r2_reg = imadjust(max(img_r2_reg, [], 3));
mip_r2_reg = imhistmatch(mip_r2_reg, mip_r2);
%%
overlay_before = zeros(size(mip_r1, 1), size(mip_r1, 2), 3);
overlay_before(:,:,1) = mip_r1 * 1.25;
overlay_before(:,:,2) = mip_r2 * 1.15;
overlay_before(:,:,3) = mip_r1 * 1.25;

overlay_after = zeros(size(mip_r1, 1), size(mip_r1, 2), 3);
overlay_after(:,:,1) = mip_r1 * 1.25;
overlay_after(:,:,2) = mip_r2_reg * 1.15;
overlay_after(:,:,3) = mip_r1 * 1.25;
%%
figure
subplot_tight(2, 4, [1 2 5 6])
imshow(overlay_after)
subplot_tight(2, 4, 3)
imshow(overlay_after(50:2000, 1000:2500, :))
subplot_tight(2, 4, 4)
imshow(overlay_after(1550:3500, 2500:4000, :))
subplot_tight(2, 4, 7)
imshow(overlay_after(2550:4500, 1000:2500, :))
subplot_tight(2, 4, 8)
imshow(overlay_after(3050:5000, 2500:4000, :))
%%
% Define ROI coordinates [y_start, y_end, x_start, x_end]
roi_coords = [
    50, 2000, 750, 2250;       % ROI 1
    1550, 3500, 2500, 4000;     % ROI 2
    2550, 4500, 750, 2250;     % ROI 3
    3550, 5500, 2500, 4000      % ROI 4
];

% Colors for each ROI box
box_colors = {'cyan', 'yellow', 'red', 'green'};

% Create figure
figure

% Main overlay with ROI boxes
subplot_tight(1, 5, 1)
imshow(overlay_after)
hold on;
for i = 1:size(roi_coords, 1)
    y_start = roi_coords(i, 1);
    y_end = roi_coords(i, 2);
    x_start = roi_coords(i, 3);
    x_end = roi_coords(i, 4);
    width = x_end - x_start;
    height = y_end - y_start;
    rectangle('Position', [x_start, y_start, width, height], ...
        'EdgeColor', box_colors{i}, 'LineWidth', 1);
end
hold off;

subplot_tight(1, 5, 2)
imshow(overlay_after(50:2000, 750:2250, :))
hold on;
plot([1 size(overlay_after(50:2000, 750:2250, :), 2) size(overlay_after(50:2000, 1000:2500, :), 2) 1 1], ...
     [1 1 size(overlay_after(50:2000, 750:2250, :), 1) size(overlay_after(50:2000, 1000:2500, :), 1) 1], ...
     'Color', box_colors{1}, 'LineWidth', 3);
hold off;

subplot_tight(1, 5, 3)
imshow(overlay_after(1550:3500, 2500:4000, :))
hold on;
plot([1 size(overlay_after(1550:3500, 2500:4000, :), 2) size(overlay_after(1550:3500, 2500:4000, :), 2) 1 1], ...
     [1 1 size(overlay_after(1550:3500, 2500:4000, :), 1) size(overlay_after(1550:3500, 2500:4000, :), 1) 1], ...
     'Color', box_colors{2}, 'LineWidth', 3);
hold off;

subplot_tight(1, 5, 4)
imshow(overlay_after(2550:4500, 750:2250, :))
hold on;
plot([1 size(overlay_after(2550:4500, 750:2250, :), 2) size(overlay_after(2550:4500, 1000:2500, :), 2) 1 1], ...
     [1 1 size(overlay_after(2550:4500, 750:2250, :), 1) size(overlay_after(2550:4500, 1000:2500, :), 1) 1], ...
     'Color', box_colors{3}, 'LineWidth', 3);
hold off;

subplot_tight(1, 5, 5)
imshow(overlay_after(3550:5500, 2500:4000, :))
hold on;
plot([1 size(overlay_after(3550:5500, 2500:4000, :), 2) size(overlay_after(3050:5000, 2500:4000, :), 2) 1 1], ...
     [1 1 size(overlay_after(3550:5500, 2500:4000, :), 1) size(overlay_after(3050:5000, 2500:4000, :), 1) 1], ...
     'Color', box_colors{4}, 'LineWidth', 3);
hold off;

exportgraphics(gcf, "~/Desktop/abbott.png", "Resolution",300);