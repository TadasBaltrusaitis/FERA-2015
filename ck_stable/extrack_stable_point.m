ck_point_dir = 'F:\Dropbox\Dropbox\ActionUnits\fera2015\ck_points\';

point_files = dir([ck_point_dir, '*.txt']);

stds_x = zeros(numel(point_files), 68);
stds_y = zeros(numel(point_files), 68);

stds_eucl = zeros(numel(point_files), 68);

for i = 1:numel(point_files)
    
   points = dlmread([ck_point_dir, point_files(i).name], ' ');
    
   valid = points(:,2);
   points = points(:, 3:2:end);
   
   xs = points(:,1:68);
   ys = points(:,69:end);
   
   stds_x(i,:) = std(xs);
   stds_y(i,:) = std(ys); 
   
   % assume first frame neutral (it should be in CK)
   xs_diff = bsxfun(@plus, xs, -xs(1,:));
   ys_diff = bsxfun(@plus, ys, -ys(1,:));
      
   eucl_dists = mean(sqrt(xs_diff.^2 + ys_diff.^2));
   stds_eucl(i,:) = eucl_dists;
end
inds = 1:68;

inds(mean(stds_eucl) < 2.6)