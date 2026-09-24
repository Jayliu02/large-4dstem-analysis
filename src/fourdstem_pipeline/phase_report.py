"""Chinese reports and measured-pattern evidence for phase identification."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .basic_analysis import write_report
from .loaders import load_dataset
from .phase_peaks import atomic_json, read_json


def transformed_predictions(library, result, scale, center, visibility, inner, outer):
    index = int(result[4])
    if index < 0:
        return np.empty((0,2)), np.empty((0,3), dtype=int)
    n = int(library['count'][index])
    points = library['qxy'][index,:n].copy()/scale
    points[:,1] *= result[5]
    angle = float(result[3])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    relative = points @ rotation.T
    points = relative+center
    radii = np.linalg.norm(relative,axis=1)
    pixels = np.rint(points).astype(int)
    keep = ((radii >= inner) & (radii <= outer) & (pixels[:,0] >= 4) & (pixels[:,0] < visibility.shape[0]-4)
            & (pixels[:,1] >= 4) & (pixels[:,1] < visibility.shape[1]-4))
    inside = np.flatnonzero(keep)
    keep[inside] &= visibility[pixels[inside,0],pixels[inside,1]]
    return points[keep], library['hkl'][index,:n][keep]


def evidence_pairs(observed, predicted, tolerance):
    distance = np.linalg.norm(observed[:,None,:]-predicted[None,:,:],axis=-1)
    rows,cols = np.where(distance <= tolerance)
    order = np.argsort(distance[rows,cols])
    used_i,used_j,pairs = set(),set(),[]
    for k in order:
        i,j = int(rows[k]),int(cols[k])
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            pairs.append((i,j,float(distance[i,j])))
    return pairs


def scan_report(path, basic, output, peaks, arrays, libraries, calibration, cfg, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    candidates = cfg['candidates']
    phase_ids = [-2, -1] + [c['id'] for c in candidates]
    names = ['ambiguous', 'unindexed'] + [c['label'] for c in candidates]
    labels = dict(zip(phase_ids, names))
    palette = plt.get_cmap('tab10')
    # Keep the default accepted phases blue/green, distinct from amber ambiguity.
    palette_order = [0, 2, 4, 3, 5, 6, 8, 9, 1, 7]
    colors = ListedColormap(['#e5a83c', '#cccccc'] + [palette(palette_order[i % 10]) for i in range(len(candidates))])
    norm = BoundaryNorm(np.arange(-2.5, len(candidates) + .5), colors.N)
    figures = []
    fig,axes = plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,key,title in [(axes[0,0],'phase_id','Accepted phase (all checks)'),
                         (axes[0,1],'best_candidate','Best candidate (includes rejected fits)')]:
        artist=ax.imshow(arrays[key],cmap=colors,norm=norm)
        bar=fig.colorbar(artist,ax=ax,ticks=phase_ids,shrink=.8)
        bar.ax.set_yticklabels(names)
        ax.set_title(title)
    for ax,key,title in [(axes[0,2],'score','Minimum winning score across voltages'),
                         (axes[1,0],'margin','Minimum normalized phase margin'),
                         (axes[1,1],'matched_peaks','Matched spots (best fit)'),
                         (axes[1,2],'median_residual_px','Median residual (best fit, px)')]:
        data=np.asarray(arrays[key]).copy().astype(float)
        data[~np.isfinite(data)]=np.nan
        artist=ax.imshow(data,cmap='viridis')
        fig.colorbar(artist,ax=ax,shrink=.8)
        ax.set_title(title)
    for ax in axes.ravel():
        ax.set(xlabel='scan x',ylabel='scan y')
    fig.savefig(output/'phase_maps.png',dpi=150)
    plt.close(fig)
    figures.append(('相标签与匹配质量。最佳候选图包含未通过检验的位置，不能作为相分布图解读。','phase_maps.png'))

    motion=read_json(basic/'beam_motion.json')
    yy,xx=np.indices(peaks['count'].shape)
    predicted=np.stack([yy,xx],axis=-1) @ np.asarray(motion['affine_coefficients_yx']).T+motion['affine_intercept_yx']
    residual=peaks['center_yx']-predicted
    fig,axes=plt.subplots(1,4,figsize=(17,4),constrained_layout=True)
    for ax,data,title in zip(axes,[peaks['center_yx'][...,0],peaks['center_yx'][...,1],np.linalg.norm(residual,axis=-1),peaks['center_valid']],
                             ['Measured beam y (px)','Measured beam x (px)','Residual to affine prediction (px)','Accepted beam center']):
        artist=ax.imshow(data)
        ax.set_title(title)
        fig.colorbar(artist,ax=ax,shrink=.8)
    fig.savefig(output/'beam_centers.png',dpi=150)
    plt.close(fig)
    figures.append(('逐点束心与仿射初值的残差。匹配使用相对束心的斑点坐标，不对原始衍射图插值。','beam_centers.png'))

    samples=np.load(output/'calibration_samples.npz')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    axes[0].plot(samples['scales'],samples['training_scores'].mean(axis=1),'o-',ms=3)
    axes[0].axvline(calibration['scale_inv_angstrom_per_pixel'],color='red',label='selected working estimate')
    axes[0].axvspan(*calibration['bootstrap_interval_95'],alpha=.2,color='red')
    axes[0].set(xlabel='reciprocal scale (1/Angstrom/pixel)',ylabel='mean training agreement',title=calibration['status'])
    axes[0].legend(fontsize=8)
    axes[1].hist(samples['null_max_scores'],bins=20,label='randomized angles',alpha=.7)
    axes[1].axvline(calibration['null_score_threshold'],color='red',label='acceptance threshold')
    axes[1].set(xlabel='maximum score over phases and voltages',ylabel='patterns')
    axes[1].legend(fontsize=8)
    fig.savefig(output/'calibration.png',dpi=150)
    plt.close(fig)
    figures.append(('标定曲线、训练样本自助抽样区间与随机角度对照；尺度由全部候选相共同竞争确定。','calibration.png'))

    metadata=read_json(output/'peaks'/'input_metadata.json')
    dataset=load_dataset(path,backend='mib_memmap',scan_shape=peaks['count'].shape,
                         detector_shape=metadata['detector_shape'],dtype='>u2',mib_header_bytes=384)
    visibility=np.load(output/'detector_visibility.npy')
    representatives=[]
    for phase in [c['id'] for c in candidates]:
        mask=(arrays['phase_id']==phase)
        if not mask.any():
            mask=(arrays['best_candidate']==phase)&peaks['center_valid']
            supported=mask&(peaks['count']>=cfg['matching']['min_peaks'])&(arrays['matched_peaks']>=cfg['matching']['min_peaks'])
            if supported.any():
                mask=supported
        indices=np.flatnonzero(mask)
        if not len(indices):
            continue
        ranked=indices[np.argsort(arrays['score'].ravel()[indices])]
        for index in dict.fromkeys([int(ranked[-1]),int(ranked[len(ranked)//2])]):
            representatives.append(np.unravel_index(index,mask.shape))
    evidence=[]
    if representatives:
        columns = min(3, len(candidates))
        rows = int(np.ceil(len(representatives) / columns))
        fig,axes=plt.subplots(rows,columns,figsize=(6*columns,5*rows),constrained_layout=True,squeeze=False)
        for ax,(y,x) in zip(axes.ravel(),representatives):
            result=arrays['all_results'][y,x]
            lib_index=int(result[:,0].argmax())
            best,lib=result[lib_index],libraries[lib_index]
            obs=peaks['peak_yx'][y,x,:int(peaks['count'][y,x])]
            center=peaks['center_yx'][y,x]
            pred,hkl=transformed_predictions(lib,best,calibration['scale_inv_angstrom_per_pixel'],center,visibility,
                                             cfg['peaks']['exclusion_radius'],cfg['peaks']['max_radius'])
            pairs=evidence_pairs(obs,pred,cfg['matching']['match_radius_px'])
            raw=np.asarray(dataset.data[y,x,:,:])
            ax.imshow(np.log1p(raw),cmap='gray',vmin=0,vmax=np.percentile(np.log1p(raw),99.8))
            ax.scatter(obs[:,1],obs[:,0],s=35,facecolors='none',edgecolors='#ffc857',label='observed')
            ax.scatter(pred[:,1],pred[:,0],s=35,marker='+',color='#47cbff',label='predicted')
            ax.scatter(center[1],center[0],marker='x',color='red',label='beam')
            for i,j,distance in pairs[:12]:
                ax.plot([obs[i,1],pred[j,1]],[obs[i,0],pred[j,0]],color='#ff67d0',lw=1)
                ax.annotate(str(tuple(int(v) for v in hkl[j])),(pred[j,1],pred[j,0]),fontsize=6,color='#47cbff')
            ax.set_title(f"({y},{x}) candidate={labels[int(lib['phase_id'])]}; accepted={labels[int(arrays['phase_id'][y,x])]}\n"
                         f"{int(lib['voltage_kv'])} kV hypothesis; score={best[0]:.3f}; matches={int(best[1])}",fontsize=10)
            ax.legend(fontsize=6,loc='upper left')
            evidence.append({'scan_yx':[int(y),int(x)],'accepted_phase':int(arrays['phase_id'][y,x]),
                             'accepted_phase_label':labels[int(arrays['phase_id'][y,x])],
                             'candidate':int(lib['phase_id']),'candidate_label':labels[int(lib['phase_id'])],
                             'voltage_hypothesis_kv':int(lib['voltage_kv']),
                             'template_index':int(best[4]),'score':float(best[0]),
                             'reason_flags':int(arrays['reason_flags'][y,x]),'selected_peaks':int(peaks['count'][y,x]),
                             'matched_peaks':len(pairs),'observed_yx':obs,'predicted_yx':pred,'predicted_hkl':hkl,
                             'pairs_observed_predicted_residual':pairs})
        for ax in axes.ravel()[len(representatives):]:
            ax.set_visible(False)
        fig.savefig(output/'pattern_evidence.png',dpi=160)
        plt.close(fig)
        figures.append(('实测衍射图与最佳运动学模板叠加。黄圈为检测斑点，蓝色十字为预测反射；标题同时显示候选与最终标签。','pattern_evidence.png'))
    atomic_json(output/'representative_evidence.json',evidence)
    counts=summary['phase_counts']
    reason_zh={'scale optimum is at the search boundary':'最优尺度位于搜索边界',
               'competing scale separated by >5% has comparable score':'相差超过 5% 的其他尺度具有接近的分数',
               'bootstrap reciprocal-scale interval is wider than 4%':'尺度自助抽样区间宽度超过 4%',
               'too few held-out patterns support the fitted scale':'独立留出图样中支持该尺度的数量不足',
               'too few valid spatial samples for scale fitting and independent validation':'可用于拟合及独立验证的有效空间样本不足'}
    paragraphs=[f"输入：{path.name}。完整处理 {summary['patterns']:,} 个扫描位置；未对相标签平滑或插值。",
        '相标签：'+'，'.join(f"{c['label']}={counts[str(c['id'])]}" for c in candidates)+f"；歧义={counts['-2']}，未索引={counts['-1']}。这些比例是通过当前阈值的位置比例，不是材料体积分数。",
        f"尺度状态：{calibration['status']}；工作估计 {calibration['scale_inv_angstrom_per_pixel']:.7f} Å⁻¹/像素。"+
        ('未通过原因：'+'；'.join(reason_zh.get(r,r) for r in calibration['reasons']) if calibration['reasons'] else '通过候选结构条件下的空间留出验证；仍需外部标准样品校准。'),
        '候选结构：'+'；'.join(f"{c['label']}，CIF={c['cif']}，预期空间群={c['expected_space_group']}" for c in candidates)+'。晶格常数及原子坐标采用 CIF 原值；程序检查元素、占位和展开结构的空间群，来源指纹见结构审计。',
        f"每份扫描共用一个倒空间尺度；{'/'.join(str(v) for v in cfg['templates']['voltages_kv'])} kV 是未知电压的假设，不能从本结果宣称仪器电压。真实电压若不在此范围，稳定性检验不适用。",
        f"接受相标签要求各电压假设一致、至少 {cfg['matching']['min_peaks']} 个非中心一对一匹配斑点、二维方向支持、残差中位数 ≤{cfg['matching']['max_median_residual_px']} 像素、归一化相间分差 ≥{cfg['matching']['min_phase_margin']:.0%}，并通过束心 ±{cfg['matching']['center_perturbation_px']} 像素及尺度 ±{cfg['matching']['scale_perturbation']:.0%} 的独立扰动检验。",
        f"分数阈值为 {calibration['null_score_threshold']:.3f}，取固定阈值与随机角度对照第 {cfg['matching']['null_percentile']} 百分位的较大者。分数不是后验概率；随机对照不能替代材料真值验证。",
        f"最终标签还要求模板解释至少 {cfg['matching']['minimum_observed_fraction']:.0%} 的入选斑点。该条件减少部分斑点偶合导致的接受，仍不能证明不存在少量重叠晶粒。",
        '匹配使用完整晶胞的运动学结构因子与激发误差。模板离散、动态衍射、重叠晶粒、应变、探测器畸变以及未列入的相都可能导致拒绝或误配；本流程不输出定量取向或应变结论。',
        f"检测与匹配分开：仅保留积分强度 ≥{cfg['matching']['minimum_peak_intensity']} 且 ≥各图最强非中心斑点强度的 {cfg['matching']['minimum_peak_relative_intensity']:.0%} 的候选斑点；selected_peak_indices.npy 可追溯至原检测数组。",
        '原始 MIB 只读，计数异常仅在检测副本中处理。所有数组按原扫描位置保存；reason_flags 位掩码和 array_schema.json 给出拒绝原因。']
    links=[('数值汇总','phase_summary.json'),('标定审计','calibration.json'),('数组字段与拒绝原因','array_schema.json'),
           ('逐点相标签 NPY','phase_id.npy'),('全部候选匹配 NPY','all_results.npy'),('代表点与反射证据','representative_evidence.json'),
           ('结构及来源审计','../templates/structure_audit.json'),('运行指纹与断点','matching_checkpoint.json')]
    write_report(output,f"{' / '.join(c['label'] for c in candidates)} 相识别：{output.name}",paragraphs,figures,links)


def batch_report(root):
    import csv
    summaries=[read_json(path) for path in sorted(root.glob('scan_*/phase_summary.json'))]
    if not summaries:
        raise ValueError('No completed scan summaries found.')
    labels = summaries[0]['phase_labels']
    if any(s['phase_labels'] != labels for s in summaries):
        raise ValueError('Cannot combine scans with different candidate phase labels.')
    candidate_ids = sorted((k for k in labels if int(k) >= 0), key=int)
    candidate_names = [labels[k] for k in candidate_ids]
    atomic_json(root/'batch_summary.json',summaries)
    with (root/'phase_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['scan','calibration_status','scale_inv_A_per_pixel']+candidate_names+['ambiguous','unindexed'])
        for s in summaries:
            writer.writerow([s['output'],s['calibration_status'],s['scale']]+[s['phase_counts'][k] for k in candidate_ids+['-2','-1']])
    paragraphs=[f"采用 {'、'.join(candidate_names)} 候选库，逐点定位束心并竞争匹配。候选结构、倒空间尺度与加速电压的不确定性均纳入拒绝规则。",
                '标签定义：'+'；'.join(f'{k}={v}' for k,v in labels.items())+'。']
    for s in summaries:
        paragraphs.append(f"{s['output']}：{s['patterns']:,} 个位置，标定状态 {s['calibration_status']}；相标签计数 {s['phase_counts']}。")
    paragraphs.append('未索引或歧义不代表非晶，也不代表不存在候选相。请先查看标定状态，再查看通过全部检查的相标签；最佳候选图仅用于诊断。')
    write_report(root,f"{' / '.join(candidate_names)} 相识别汇总",paragraphs,[],[(s['output'],f"{s['output']}/report_zh.html") for s in summaries])
